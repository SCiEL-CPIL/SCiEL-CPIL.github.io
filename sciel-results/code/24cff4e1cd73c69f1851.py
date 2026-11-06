import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    
    # Extract parameters
    if hasattr(contract, 'K'):
        K = contract.K
        r = contract.r
        sigma = contract.sigma
        T = contract.T
        n_ex = contract.n_exercise_dates
    else:
        K = contract.get('K', 100)
        r = contract.get('r', 0.05)
        sigma = contract.get('sigma', 0.2)
        T = contract.get('T', 1.0)
        n_ex = contract.get('n_exercise_dates', 12)
    
    # Exercise times
    exercise_times = np.arange(1, n_ex + 1) * T / n_ex
    
    # Tree parameters
    N_steps = 1200 
    dt = T / N_steps
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    p_down = 1.0 - p_up
    disc = np.exp(-r * dt)
    
    S_0 = 100.0
    
    # Initialize at maturity
    j_N = np.arange(N_steps + 1)
    S_N = S_0 * (u ** j_N) * (d ** (N_steps - j_N))
    V_next = np.maximum(K - S_N, 0.0)
    
    boundaries = {}
    
    # Map exercise times to tree steps
    ex_steps = {}
    for i, t_ex in enumerate(exercise_times):
        k_ex = int(round(t_ex / dt))
        ex_steps[k_ex] = i + 1 # 1-based index
        
    # Backward Induction
    for k in range(N_steps - 1, -1, -1):
        # Check deadline
        if time.time() - start_time > 110:
            # Fallback to simple heuristic if we run out of time
            # This shouldn't happen
            break
            
        V_cont = disc * (p_up * V_next[:-1] + p_down * V_next[1:])
        
        j_curr = np.arange(k + 1)
        S_curr = S_0 * (u ** j_curr) * (d ** (k - j_curr))
        
        V_ex = np.maximum(K - S_curr, 0.0)
        
        if k in ex_steps:
            ex_idx = ex_steps[k]
            V_curr = np.maximum(V_ex, V_cont)
            
            mask = V_ex >= V_cont
            if np.any(mask):
                last_idx = np.where(mask)[0][-1]
                S_star = S_curr[last_idx]
            else:
                S_star = 0.0
            
            boundaries[ex_idx] = S_star
        else:
            V_curr = V_cont
            
        V_next = V_curr
        
    # If we broke early, fill missing boundaries with K (conservative)
    for i in range(1, n_ex + 1):
        if i not in boundaries:
            boundaries[i] = K
            
    class BermudanPolicy:
        def __init__(self, bounds, ex_times, n_ex):
            self.bounds = bounds
            self.ex_times = ex_times
            self.n_ex = n_ex
            # Pre-compute time to index mapping for speed
            self.time_to_idx = {}
            for i in range(1, n_ex + 1):
                self.time_to_idx[round(ex_times[i-1], 10)] = i
                
        def exercise(self, current_state, time):
            # Find corresponding boundary
            # Try exact match first
            t_key = round(time, 10)
            if t_key in self.time_to_idx:
                idx = self.time_to_idx[t_key]
            else:
                # Find closest
                idx = 1
                min_diff = float('inf')
                for i in range(1, self.n_ex + 1):
                    diff = abs(self.ex_times[i-1] - time)
                    if diff < min_diff:
                        min_diff = diff
                        idx = i
                        
            S_star = self.bounds[idx]
            return current_state <= S_star

    return BermudanPolicy(boundaries, exercise_times, n_ex)
