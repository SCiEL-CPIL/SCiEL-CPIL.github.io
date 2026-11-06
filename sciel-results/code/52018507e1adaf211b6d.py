import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices)
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    # 1. High-Resolution Binomial Tree for Initial Critical Prices
    # Use 200 steps per month -> 2400 total steps.
    # This provides a very accurate initial estimate of the boundary.
    steps_per_month = 200
    N = n_ex * steps_per_month 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    S0_ref = 100.0 
    
    # Initialize value array at maturity
    V = np.zeros(N + 1)
    i_vals = np.arange(N + 1)
    
    # Use log-space for stability and speed
    log_d = np.log(d)
    log_u = np.log(u)
    log_S_N = np.log(S0_ref) + N * log_d + i_vals * (log_u - log_d)
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    for j in range(N, 0, -1):
        # V_cont[i] = discount * (p * V[i] + (1-p) * V[i+1])
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        
        # S_prev[i] = S_curr[i] / d
        # Note: V corresponds to step j. S_curr corresponds to step j.
        # We are calculating values for step j-1.
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
        
        if time.time() > deadline - 60.0:
            break

    # 2. Refine using Training Paths
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12)
        
    M = paths.shape[0]
    
    # Discount factors for each step
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    def negative_value(c_prices):
        # Vectorized calculation of portfolio value given critical prices
        # We need to find the first k where S_k <= c_k
        # and S_i > c_i for all i < k.
        
        # W[:, k] = (S[:, k] <= c[k])
        W = paths <= c_prices[np.newaxis, :]
        
        exercised = np.zeros(M, dtype=bool)
        exercise_step = np.full(M, -1)
        
        for k in range(12):
            not_exercised = ~exercised
            cond = W[:, k] & not_exercised
            exercised[cond] = True
            exercise_step[cond] = k
            
            if np.all(exercised):
                break
                
        payoffs = np.zeros(M)
        if np.any(exercised):
            idx = np.arange(M)
            ex_idx = idx[exercised]
            ex_steps = exercise_step[exercised]
            
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            
            payoffs[ex_idx] = disc_ex * (K - S_ex)
            
        return -np.mean(payoffs)
        
    # Optimize each critical price independently (1D optimization)
    # This is faster and more robust than 12D Nelder-Mead.
    # The optimal policy is monotonic: c_1 >= c_2 >= ... >= c_12 (roughly, for puts)
    # Actually for puts, c_t increases as t increases? No.
    # Put: Exercise when S is low.
    # As time to expiry decreases, the option time value decreases, so we exercise at higher S.
    # So c_1 (t=1/12) should be lower than c_12 (t=1.0)?
    # Wait. At t=1.0 (maturity), we always exercise if ITM. So c_12 = K.
    # At t=1/12, we only exercise if deep ITM. So c_1 < K.
    # So c_1 <= c_2 <= ... <= c_12 = K.
    
    # Let's refine from last to first to respect monotonicity roughly, 
    # or just optimize independently and then enforce monotonicity if needed.
    # Independent 1D optimization is very fast.
    
    x0 = critical_prices.copy()
    x0 = np.clip(x0, 0, K)
    
    # Ensure c_12 is K (at maturity, always exercise if ITM)
    # Actually, at t=T, the option expires. If S < K, exercise. If S > K, don't.
    # So c_12 should be K.
    x0[11] = K
    
    # Optimize c_1 to c_11
    # We can do this in parallel or sequentially. Sequential is fine, it's fast.
    # To respect the "no-arbitrage" monotonicity (c_t is non-decreasing in t for puts),
    # we can constrain the search space.
    # c_k >= c_{k-1} and c_k <= c_{k+1} (for k < 12).
    
    # Let's just do simple 1D minimization for each.
    for k in range(11):  # 0 to 10
        # Bounds: [c_{k-1}, c_{k+1}] if available, else [0, K]
        lower = x0[k-1] if k > 0 else 0
        upper = x0[k+1] if k < 11 else K
        
        # Ensure lower <= upper
        if lower > upper:
            lower, upper = upper, lower
            
        # Define objective for this specific k
        def obj_k(c_k):
            x_temp = x0.copy()
            x_temp[k] = c_k
            return negative_value(x_temp)
            
        try:
            res = minimize_scalar(obj_k, bounds=(lower, upper), method='bounded', 
                                  options={'xatol': 1e-5})
            if res.success:
                x0[k] = res.x
        except Exception:
            pass
            
    # Final check: ensure monotonicity c_1 <= ... <= c_12
    # If not, project back? Or just trust the optimizer.
    # The optimizer might violate monotonicity if the training data is noisy.
    # Let's enforce it.
    for k in range(1, 12):
        if x0[k] < x0[k-1]:
            x0[k] = x0[k-1]
            
    # Ensure c_12 = K
    x0[11] = K
    
    return ExercisePolicy(x0)
