import numpy as np

def fit_exercise_policy(training_paths, contract):
    # Extract parameters
    # S0, K, r, sigma, T are fixed in the contract or known from task description
    # The task says: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1.
    # We should probably extract these from `contract` if available, or hardcode if `contract` is just a placeholder.
    # Let's assume `contract` contains the parameters or we use the defaults from the prompt.
    
    # The prompt says "fixed model/data assumptions".
    # "Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1"
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    # Number of exercise dates
    n_exercise = 12
    exercise_times = np.linspace(1/12.0, T, n_exercise)
    
    # Build Binomial Tree
    # Use a high number of steps for accuracy
    N = 2400 # 200 steps per month, very fine
    dt = T / N
    steps_per_month = int(round(N / n_exercise)) # 200
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # We need to store values for the final time step, then work backwards.
    # Since it's 1D, we can just keep an array of values for the current time step.
    # Max nodes at T is N+1.
    
    # Initialize values at T
    # S at T: S0 * u^j * d^(N-j) ? No, S depends on the path.
    # In a binomial tree, the price at step i, node j is S0 * u^j * d^(i-j).
    # We need to evaluate payoff at T for all nodes.
    
    # Let's create the grid of S values at T.
    # j ranges from 0 to N.
    j = np.arange(N + 1)
    S_T = np.zeros(N + 1)
    # To avoid underflow/overflow, calculate log S?
    # log S_T = log S0 + j * log u + (N-j) * log d
    # But S0 varies. However, the *boundary* S* is independent of S0?
    # No, the optimal stopping boundary for a put is a specific price level.
    # The decision "Exercise if S <= S*" is independent of the initial S0.
    # So we can just compute the boundary for a generic S.
    
    # Let's compute the continuation value and exercise value for a range of S.
    # It's easier to work with the tree values.
    # V[i, j] is value at step i, node j.
    # We only need to store V for the current step i.
    
    # At T (i=N):
    # S_N_j = S0 * u^j * d^(N-j).
    # This depends on S0. But the *condition* for exercise is S <= S*.
    # The value V(S) is homogeneous? No.
    # However, we can just compute the critical price by scanning a grid of S values.
    
    # Alternative: Just compute the tree for a specific S0, say S0=100.
    # The critical price S* will be the same regardless of S0?
    # Yes, for a 1D GBM, the optimal stopping boundary is a function of time only, $S^*(t)$.
    # So we can pick any S0 to build the tree and find the boundary.
    
    S0 = 100.0
    
    # Calculate S values at T
    # S_T[j] = S0 * (u/d)^(j - N/2) * ... no
    # S_T[j] = S0 * u^j * d^(N-j)
    # To compute this efficiently:
    # S_T[j] = S0 * exp(j * log(u) + (N-j) * log(d))
    log_u = np.log(u)
    log_d = np.log(d)
    log_S_T = np.log(S0) + j * log_u + (N - j) * log_d
    S_T = np.exp(log_S_T)
    
    V = np.maximum(K - S_T, 0.0)
    
    # Backward induction
    # We need to check if time i*dt is an exercise date.
    # Exercise dates are at k * dt_month where dt_month = T/12.
    # In the tree, step i corresponds to time i*dt.
    # i*dt = k * T/12  => i = k * N/12.
    # So we exercise at steps i = 0, N/12, 2N/12, ..., N.
    # Note: i=0 is t=0, but problem says "no time-0 exercise".
    # Exercise dates: 1/12, 2/12, ..., 12/12.
    # So i = N/12, 2N/12, ..., N.
    
    exercise_steps = set(int(round(k * N / 12.0)) for k in range(1, 13))
    
    # We will store the critical prices for each exercise date.
    # critical_prices[k] for k=1..12.
    critical_prices = [0.0] * 13 # index 1 to 12
    
    # We need to track the S values at each step to find the boundary.
    # Instead of storing the whole tree, we can just find the boundary at each exercise step.
    # At step i, the S values are S_i[j].
    # We need to find the largest j such that Exercise >= Continuation.
    # Since S decreases as j decreases (assuming u>1>d), the condition S <= S* means small j.
    # So we look for the boundary between "Exercise" and "Continue".
    
    # Let's iterate backwards.
    for i in range(N - 1, -1, -1):
        # Calculate S values at step i
        # S_i[j] = S0 * u^j * d^(i-j)
        # j ranges from 0 to i.
        j_i = np.arange(i + 1)
        log_S_i = np.log(S0) + j_i * log_u + (i - j_i) * log_d
        S_i = np.exp(log_S_i)
        
        # Calculate Continuation Value
        # V_cont = exp(-r dt) * (p * V[j+1] + (1-p) * V[j])
        # V currently holds values for step i+1, indices 0..i+1.
        # V[j+1] corresponds to up-move, V[j] to down-move.
        V_up = V[1:] # size i
        V_down = V[:-1] # size i
        # Wait, V has size i+2 (indices 0..i+1).
        # V_up should be V[1..i+1], size i+1.
        # V_down should be V[0..i], size i+1.
        V_cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_down)
        
        V_ex = np.maximum(K - S_i, 0.0)
        
        # Check if this step is an exercise step
        if i in exercise_steps:
            # Find the boundary
            # We want the largest S such that V_ex >= V_cont.
            # S_i is increasing with j.
            # So we want the largest j such that V_ex[j] >= V_cont[j].
            # Then S* = S_i[j_max].
            
            # Handle case where all are exercised or none are exercised
            diff = V_ex - V_cont
            # Find indices where diff >= 0
            valid = np.where(diff >= 0)[0]
            if len(valid) > 0:
                j_max = valid[-1]
                critical_price = S_i[j_max]
            else:
                # If no exercise is optimal, critical price is 0 (never exercise)
                # Or if all exercise, it's the max S.
                # If diff is always < 0, we never exercise. S* = 0.
                critical_price = 0.0
            
            # Map step i to exercise index k
            k = int(round(i * 12.0 / N))
            # Ensure k is within 1..12
            if 1 <= k <= 12:
                critical_prices[k] = critical_price
            
            # Update V: at exercise step, we take max
            V = np.maximum(V_cont, V_ex)
        else:
            # No exercise, only continuation
            V = V_cont
            
    # The critical_prices array now holds the boundaries.
    # Note: Due to floating point, we might want to smooth or ensure monotonicity?
    # The boundary should be non-decreasing as time approaches T?
    # For a put, as T approaches, the boundary increases towards K.
    # So S*_1 <= S*_2 <= ... <= S*_12.
    
    # Return the policy object
    return ExercisePolicy(critical_prices)

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)
        
    def exercise(self, current_state, time):
        # time is a float
        # Determine k
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        S_star = self.critical_prices[k_int]
        return current_state <= S_star
