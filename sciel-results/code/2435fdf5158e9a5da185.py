import numpy as np
import time
from scipy.optimize import minimize

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        # current_state: array-like
        # time: float
        # Determine exercise index
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        # Validate index
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return np.asarray(current_state) <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    # 1. Get initial critical prices from a high-resolution binomial tree
    # Use 200 steps per month for a good starting point (fast)
    steps_per_month = 200
    N = n_ex * steps_per_month 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    S0_ref = 100.0 
    
    V = np.zeros(N + 1)
    i_vals = np.arange(N + 1)
    log_d = np.log(d)
    log_u = np.log(u)
    log_S_N = np.log(S0_ref) + N * log_d + i_vals * (log_u - log_d)
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    for j in range(N, 0, -1):
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        S_prev = S_curr[0:j] / d
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
            
        V = V_new
        S_curr = S_prev
        
        if time.time() > deadline - 30.0:
            break

    # 2. Optimize critical prices using training_paths
    # Assume training_paths shape is (M, 12)
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
        
    M = paths.shape[0]
    
    # Discount factors for each step
    # t_k = k/12
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    def negative_value(c_prices):
        # c_prices: array of 12 critical prices
        # Vectorized simulation
        # We need to find the first k where S_k <= c_k
        # and S_i > c_i for all i < k
        
        # Create a mask for each step
        # S is (M, 12)
        # c is (12,)
        
        # Exercise condition at step k: S[:, k] <= c[k]
        # AND not exercised at any previous step.
        
        # To do this efficiently:
        # 1. Compute "would exercise" mask: W[:, k] = (S[:, k] <= c[k])
        # 2. Compute "first exercise" index.
        
        W = paths <= c_prices[np.newaxis, :]
        
        # We want the first True in each row.
        # If no True, payoff is 0.
        
        # Trick:
        # Cumulative sum of ~W?
        # Let's use a loop over 12 steps, it's small.
        
        exercised = np.zeros(M, dtype=bool)
        exercise_step = np.full(M, -1)
        
        for k in range(12):
            # Not exercised yet
            not_exercised = ~exercised
            # Check condition at step k
            cond = W[:, k] & not_exercised
            # Update
            exercised[cond] = True
            exercise_step[cond] = k
            
            # Early break if all exercised?
            if np.all(exercised):
                break
                
        # Calculate payoff
        # For those exercised, payoff = discount[k] * (K - S[k])
        # For those not, 0.
        
        payoffs = np.zeros(M)
        if np.any(exercised):
            # We need to pick the correct S and discount factor for each path.
            # S_ex = paths[exercised, exercise_step[exercised]]
            # This indexing is tricky.
            
            # Let's use a loop for payoff calculation if vectorization is hard.
            # Or use advanced indexing.
            
            idx = np.arange(M)
            ex_idx = idx[exercised]
            ex_steps = exercise_step[exercised]
            
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            
            payoffs[ex_idx] = disc_ex * (K - S_ex)
            
        return -np.mean(payoffs)
        
    # Optimization
    try:
        # Bounds: 0 to K
        bounds = [(0, K) for _ in range(12)]
        # Initial guess
        x0 = critical_prices
        # Ensure x0 is within bounds
        x0 = np.clip(x0, 0, K)
        
        # Nelder-Mead
        res = minimize(negative_value, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < negative_value(x0):
            critical_prices = res.x
        else:
            critical_prices = x0
            
    except Exception:
        # Fallback to tree
        pass

    return ExercisePolicy(critical_prices)
