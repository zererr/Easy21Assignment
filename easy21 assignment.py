import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from pprint import pprint 
from tqdm import tqdm 
from matplotlib.patches import Patch

class Easy_21_Environment():
    def __init__(self, discount, N_0):
        self.state = None
        self.reward = None                
        self.discount = discount
        self.N_0 = N_0
        self.action_space = ['stick', 'hit']

        # Model free initialization
        self.N_a = defaultdict(lambda: 0) # Key: (s,a), value: number of times (s,a) visited
        self.N_s = defaultdict(lambda: 0) # Key: s, value: number of times s visited
        self.Q = defaultdict(lambda: 0) # Key: s, value: state value at s
        self.policy = defaultdict(lambda: random.choice(self.action_space)) # Key: s, value: randomly chosen action
              
    def reset(self, debug = False):
        self.state = tuple(int(x) for x in np.random.randint(1, 11, 2)) # Both players draw a black card
        self.dealer_sum = self.state[0] # Agent can only view dealer's first drawn card, so use this variable instead of the state
        self.reward = 0 # Episode starts without reward
        self.terminated = False

        # Debug
        if debug:
            print(f"Agent starts with {self.state[1]}, Dealer starts with {self.state[0]}")
        return self.state

    def draw(self, agent = True, debug = False):
        drawn = np.random.randint(1,11) * int(np.random.choice([1,-1], p = [2/3, 1/3]))
            
        if not agent: # Dealer draws
            self.dealer_sum += drawn            
        else: # Agent draws            
            self.state = (self.state[0], self.state[1] + drawn)
        
        # Debug
        if debug: 
            if agent:
                print(f'Agent draws {drawn}')
            else:
                print(f'Dealer draws {drawn}')
    
    def step(self, action, debug = False):
        if action == 'stick': # Agent will never be bust here, since you can only bust after a hit
            if debug:
                print(f"Agent sticks with {self.state[1]}")

            while self.dealer_sum < 17: # Dealer hits
                self.draw(agent = False, debug = debug)

            # Check if dealer busts, else compare agent and dealer's sum   
            if self.dealer_sum > 21 or self.dealer_sum < 1:
                self.reward = 1
            else: # 17 <= Dealer's sum <= 21
                if self.dealer_sum > self.state[1]:
                    self.reward = -1
                elif self.dealer_sum < self.state[1]:
                    self.reward = 1
                else:
                    self.reward = 0
            self.terminated = True
            
        else: # action == 'hit'
            self.draw(agent = True, debug = debug)
            if self.state[1] > 21 or self.state[1] < 1: # If agent busts, it always lose regardless of dealer's hand
                self.reward = -1
                self.terminated = True
            else:
                self.reward = 0

        return self.state, self.reward            
                
    def sample_episode(self, debug = False):
        state = self.reset(debug)
        trajectory = []

        while not self.terminated:
            action = self.policy[self.state]

            next_state, reward = self.step(action, debug)

            trajectory.append((state, action, reward))

            state = next_state

        if debug:
            print(f'Agent hand: {self.state[1]}, Dealer hand: {self.dealer_sum}')
            print(f'Total reward: {self.reward}')
            outcomes = {0:'Tie', 1:'Agent wins', -1: 'Dealer wins'}
            print(f"{outcomes[self.reward]}")
        return trajectory

    def evaluate_agent(self, n):        
        rewards = np.zeros(n)
        n_wins = 0
        n_ties = 0
        n_losses = 0
        
        for i in range(n):
            trajectory = self.sample_episode(debug = False)
            rewards[i] = trajectory[-1][2]
            if rewards[i] == 1:
                n_wins += 1
            elif rewards[i] == 0:
                n_ties += 1 
            else:
                n_losses += 1
        
        print(f"Win percentage ({n=}): {round(n_wins / n * 100, 2)}%")
        print(f"Tie percentage ({n=}): {round(n_ties / n * 100, 2)}%")
        print(f"Loss percentage ({n=}): {round(n_losses / n * 100, 2)}%")

    def MC_policy_iteration(self, n_episodes):
        for _ in tqdm(range(n_episodes)):
            trajectory = self.sample_episode(debug = False)        
            self.MC_eval(trajectory)
            self.epsilon_greedy_control(0, uniform = False)                

    def MC_eval(self, trajectory):
        ret = 0
        for state, action, reward in reversed(trajectory):
            self.N_s[state] += 1                
            self.N_a[(state, action)] += 1  
            ret = ret * self.discount + reward

            alpha = 1/self.N_a[state, action]

            self.Q[(state, action)] += alpha * (ret - self.Q[(state, action)])

    def epsilon_greedy_control(self, epsilon, uniform = True, tabular = True):        
        for s in self.N_s.keys():
            # If uniform is False, then use GLIE
            if not uniform:
                epsilon = self.N_0 / (self.N_0 + self.N_s[s])            
            
            if np.random.uniform() < epsilon: # Take any action uniformly
                self.policy[s] = random.choice(self.action_space)
            else: # Take the greedy action
                if tabular:            
                    self.policy[s] = max(self.action_space, key=lambda a: self.Q[(s, a)])
                else: #FA
                    self.policy[s] = max(self.action_space, key=lambda a: self.Q(s, a))

    def sarsa_lambda_forward_tabular(self, trajectory, λ):
        """
        Forward-view SARSA(λ) for a single episode.

        trajectory: list of (state, action, reward)
                    reward is the reward obtained AFTER taking the action in state
        λ: trace decay parameter (0 ≤ λ ≤ 1)
        """

        T = len(trajectory)

        for t, (state_t, action_t, _) in enumerate(trajectory):
            self.N_s[state_t] += 1
            self.N_a[(state_t, action_t)] += 1
            alpha = 1 / self.N_a[(state_t, action_t)]

            # Reset λ-return for this state–action
            λ_return = 0.0

            # Step 1: Add the possible n-step returns
            for n in range(1, T - t):
                G_n = 0.0

                # 1a) Sum rewards from step t to t+n-1
                for k in range(n):
                    r_k = trajectory[t + k][2]  # reward at step t+k
                    G_n += (self.discount ** k) * r_k

                # 1b) Bootstrapping with value estimate at t+n
                s_next, a_next, _ = trajectory[t + n]
                G_n += (self.discount ** n) * self.Q[(s_next, a_next)]

                λ_return += (1 - λ) * (λ ** (n - 1)) * G_n

            # Step 2: Add the Monte Carlo tail term
            G_T = 0.0
            for k in range(T - t):
                r_k = trajectory[t + k][2]
                G_T += (self.discount ** k) * r_k
            λ_return += (λ ** (T - t - 1)) * G_T

            # Step 3: Update Q
            self.Q[(state_t, action_t)] += alpha * (λ_return - self.Q[(state_t, action_t)])

        return self.Q

    def sarsa_lambda_forward_FA(self, trajectory, λ, alpha):
        T = len(trajectory)

        for t, (state_t, action_t, _) in enumerate(trajectory):
            self.N_s[state_t] += 1                
            self.N_a[(state_t, action_t)] += 1

            # Reset λ-return for this state–action
            λ_return = 0.0

            # Step 1: Add the possible n-step returns
            for n in range(1, T - t):
                G_n = 0.0

                # 1a) Sum rewards from step t to t+n-1
                for k in range(n):
                    r_k = trajectory[t + k][2]  # reward at step t+k
                    G_n += (self.discount ** k) * r_k

                # 1b) Bootstrapping with value estimate at t+n
                if t + n < T:
                    s_next, a_next, _ = trajectory[t + n]
                    G_n += (self.discount ** n) * self.Q(s_next, a_next) # Must use Q_hat here

                λ_return += (1 - λ) * (λ ** (n - 1)) * G_n

            # Step 2: Add the Monte Carlo tail term
            G_T = 0.0
            for k in range(T - t):
                r_k = trajectory[t + k][2]
                G_T += (self.discount ** k) * r_k
            λ_return += (λ ** (T - t - 1)) * G_T

            # Step 3: Update weights and construct new Q
            self.weights += alpha * (λ_return - self.Q(state_t, action_t)) * feature_vector(state_t, action_t)
            self.Q = linear_FA_constructor(feature_vector, self.weights)

# Helper functions

def MSE(Q, Q_star, default_value=0.0, tabular = True):
    # Computes the mean squared error between a given action value function Q, and the optimal action value function Q_star    
    # If Q is a function and hence not tabular, use the keys of Q_star
    SE = 0

    if tabular:
        all_keys = set(Q.keys()) | set(Q_star.keys())
        for key in all_keys:
            v1 = Q.get(key, default_value)
            v2 = Q_star.get(key, default_value)
            SE += (v1 - v2) ** 2

    else: # FA, Q = Q(s,a,w)
        all_keys = set(Q_star.keys())
        for key in all_keys:
            v1 = Q(key[0], key[1])
            v2 = Q_star.get(key)
            SE += (v1 - v2)**2    
    
    return SE / len(all_keys) # Mean SE

def compute_optimal_V(Q):
    # Only for tabular Q. Computes V* = max_a(Q*(s,a))
    V = {}
    for (s, a), q_val in Q.items():
        if s not in V:
            V[s] = q_val
        else:
            V[s] = max(V[s], q_val)
    return V

def within(a,b):
    # Helper function for constructing feature vector used in functional approximation
    return lambda x: a <= x <= b

def feature_vector(s, a):
    # Feature vector for coarse coding. See assignment details for explanation.
    f1 = [within(1,4), within(4,7), within(7,10)]
    f2 = [within(1,6), within(4,9), within(7,12), within(10,15), within(13,18), within(16,21)]
    f3 = [lambda x: x == 'hit', lambda x: x == 'stick']

    ϕ = np.zeros(len(f1) * len(f2) * len(f3))
    i = 0

    for pred1 in f1:
        for pred2 in f2:
            for pred3 in f3:
                ϕ[i] = pred1(s[0]) * pred2(s[1]) * pred3(a)
                i+=1
    return ϕ

def linear_FA_constructor(feature, weights):
    # Used to construct a new approximate value function whenever weights are updated by the learning algorithm
    def linear_FA(s, a):
        return np.dot(feature(s,a), weights)
    return linear_FA


# Plotting functions
def plot_optimal_Q(Q, title):
    # Plots the optimal state value function against the state variables (1. value of dealer's first card, 2. sum of players cards)    

    # Compute V*
    V = compute_optimal_V(Q) # Only for tabular Q

    # Plotting
    dealer_space = range(1, 11) # x, i
    player_space = range(1, 22) # y, j
    
    V_matrix = np.zeros((len(dealer_space), len(player_space))) # 10 x 21

    for i, dealer_show in enumerate(dealer_space):
        for j, player_sum in enumerate(player_space):
            state = (dealer_show, player_sum)
            V_matrix[i,j] = V.get(state, 0)  

    X, Y = np.meshgrid(player_space, dealer_space) # X and Y are 10 x 21

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, V_matrix, cmap='viridis') # We need X, Y and V_matrix to match in shapes

    ax.set_xlabel("Player Sum")
    ax.set_ylabel("Dealer Showing")    
    ax.set_zlabel("Value V*")

    ax.set_xlim(1,21)
    ax.set_ylim(1,10)    
    ax.set_zlim(-1,1)

    ax.set_xticks(range(1,22,2))
    ax.set_yticks(range(1,11))
    ax.set_zticks(np.arange(-1,1.2,0.2))
    
    ax.set_title(title)

    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.show()    

def plot_policy(policy, title):
    # Plots the policy of an agent

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = {'stick': 'skyblue', 'hit': 'lightcoral'}

    # Sort first, because 
    dealer_showing = sorted(set(k[0] for k in policy.keys()))
    player_sum = sorted(set(k[1] for k in policy.keys()))

    for (dealer, player), action in policy.items():
        ax.add_patch(plt.Rectangle((player_sum.index(player), dealer_showing.index(dealer)), 1, 1, color=cmap[action]))

    ax.set_xticks(np.arange(len(player_sum)) + 0.5)
    ax.set_yticks(np.arange(len(dealer_showing)) + 0.5)
    ax.set_xticklabels(player_sum)
    ax.set_yticklabels(dealer_showing)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_title(title)

    # Add legend for actions
    legend_elements = [Patch(facecolor='skyblue', label='Stick'), 
                       Patch(facecolor='lightcoral', label='Hit'),
                       Patch(facecolor='white', label='No action')]
    fig.legend(handles=legend_elements, loc='upper right', title='Action')

    # Set limits and invert y-axis for a matrix like view
    ax.set_xlim(0, len(player_sum))
    ax.set_ylim(0, len(dealer_showing))
    ax.invert_yaxis()

    plt.show()

def load_optimal_agent():
    # Constructs a Monte Carlo optimal agent object
    with open('Easy21OptimalQ.pkl', 'rb') as f:
        Q_star = pickle.load(f)
    with open('Easy21OptimalPolicy.pkl', 'rb') as f:
        pi_star = pickle.load(f)

    Optimal_Agent = Easy_21_Environment(1, 100)
    Optimal_Agent.Q = Q_star
    Optimal_Agent.policy = pi_star

    return Optimal_Agent

def plot_learning_curve(MSEs, lambdas, title):
    # Plots MSE against episode number for given lambdas
    fig, ax = plt.subplots(figsize = (10,6))

    for i, λ in enumerate(lambdas):
        ax.plot(MSEs[:, i], label=r'$\lambda$ = ' + str(round(λ, 2)))
        ax.set_xlabel('Episode')
        ax.set_ylabel('MSE')

    ax.set_title(title)    
    fig.legend()
    fig.tight_layout()

    plt.show()

def plot_MSE_against_lambda(lambdas, MSEs, title):    
    plt.plot(lambdas, MSEs)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE(Q, Q*)')
    plt.title(title)
    plt.show()

# Experiment functions
def MC_Experiment(n_episodes):
    # Used for learning Q_star, n_episodes >= 10^6 for optimal policy
    env = Easy_21_Environment(discount = 1, N_0 = 100)
    env.MC_policy_iteration(n_episodes)
    plot_optimal_Q(env.Q, title = f"Optimal State Value Function, n_episodes = {n_episodes}")
    plot_policy(env.policy, title = f'Optimal Policy Visualization, n_episodes = {n_episodes}')

    return env

def Sarsa_λ_Tabular_Experiment(lambdas, n_episodes, Q_star):
    MSEs = np.zeros((n_episodes, len(lambdas))) # For plotting learning curves of \lambda = 0 and 1
    envs = []    

    for i, λ in enumerate(lambdas):
        env.λ = λ
        env = Easy_21_Environment(discount = 1, N_0 = 100) # Remember to initialize an env for other learning algorithms

        # Training loop
        for j in tqdm(range(n_episodes)):
            trajectory = env.sample_episode(debug = False)
            Q = env.sarsa_lambda_forward_tabular(trajectory, λ)
            MSEs[j, i] = MSE(Q, Q_star)            
            env.epsilon_greedy_control(0, uniform = False, tabular = True)

        # Saving for analysis
        final_MSEs = MSEs[-1, :] # For plotting MSE against \lambda
        envs.append(env)

    # Plot results    
    plot_learning_curve(MSEs, lambdas, title = r"Learning curves for Tabular Sarsa $\lambda$")
    plot_MSE_against_lambda(lambdas, final_MSEs,
                            title = r'MSE between $Q^*$ and Q learned from Tabular Sarsa $\lambda$ after ' + f'{n_episodes} episodes')    
    for env in envs:
        plot_policy(env.policy, title = f"Policy Visualization \lambda = {env.λ}")

    return envs

def Sarsa_λ_FA_Experiment(lambdas, n_episodes, Q_star):
    MSEs = np.zeros((n_episodes, len(lambdas))) # For plotting learning curves of \lambda = 0 and 1
    envs = []

    for i, λ in enumerate(lambdas):
        env.λ = λ
        env = Easy_21_Environment(discount = 1, N_0 = None) # Remember to initialize an env for other learning algorithms
        env.weights = np.zeros(36) # 6 x 3 x 2 features, which forms a feature vector that is dot product with this weights vector
        env.Q = linear_FA_constructor(feature_vector, env.weights)

        # Training loop
        for j in tqdm(range(n_episodes)):
            trajectory = env.sample_episode(debug=False)
            env.sarsa_lambda_forward_FA(trajectory, λ, 0.01)

            MSEs[j, i] = MSE(env.Q, Q_star, tabular = False)   
            env.epsilon_greedy_control(0.05, uniform = True, tabular = False)

        # Saving for analysis
        final_MSEs = MSEs[-1, :] # For plotting MSE against \lambda
        envs.append(env)

    # Plot results    
    plot_learning_curve(MSEs, lambdas, title = r"Learning curves for FA Sarsa $\lambda$")
    plot_MSE_against_lambda(lambdas, final_MSEs,
                            title = r'MSE between $Q^*$ and Q learned from FA Sarsa $\lambda$ after ' + f'{n_episodes} episodes')
    for env in envs:
        plot_policy(env.policy, title = f"Policy Visualization \lambda = {env.λ}")

    return envs

if __name__ == '__main__':    
    Optimal_Agent = load_optimal_agent()
        
    # Initialize parameters
    n_episodes = 1000
    lambdas = np.arange(0, 1.1, 0.1)

    # Choose experiment
    env = MC_Experiment(n_episodes)
    
    # envs = Sarsa_λ_FA_Experiment(lambdas, n_episodes, Optimal_Agent.Q)

    # envs = Sarsa_λ_Tabular_Experiment(lambdas, n_episodes, Optimal_Agent.Q)
    
    