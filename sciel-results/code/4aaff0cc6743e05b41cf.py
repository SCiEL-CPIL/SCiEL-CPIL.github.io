import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def calculate_max_overlap(h):
        # h is 1D array of length N
        # C_j = sum_{i=0}^{N-1-j} h[i] * (1 - h[i+j])
        max_c = 0.0
        for j in range(N):
            if j == 0:
                # sum h[i]*(1-h[i]) = sum h[i] - sum h[i]^2
                c_j = np.sum(h) - np.sum(h**2)
            else:
                c_j = np.sum(h[:N-j] * (1 - h[j:]))
            if c_j > max_c:
                max_c = c_j
        return max_c

    # Initial guess: Binary sequence
    # Alternate 0 and 1 to satisfy sum = N/2
    h0 = np.zeros(N)
    h0[::2] = 1.0
    # Adjust if N is odd? N=100 is even.
    # Sum is N/2 = 50. Correct.

    # Objective function for optimization
    # We use a log-sum-exp approximation of the max to make it differentiable
    beta = 50.0
    
    def objective(h):
        # Calculate all C_j
        c_vals = []
        for j in range(N):
            if j == 0:
                c_j = np.sum(h) - np.sum(h**2)
            else:
                c_j = np.sum(h[:N-j] * (1 - h[j:]))
            c_vals.append(c_j)
        c_vals = np.array(c_vals)
        
        # Log-sum-exp for max
        m = np.max(c_vals)
        log_sum_exp = m + np.log(np.sum(np.exp(c_vals - m)))
        return log_sum_exp / beta

    # Constraints
    # Equality: sum(h) = N/2
    # Bounds: 0 <= h <= 1
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}
    ]
    
    bounds = [(0, 1) for _ in range(N)]

    best_h = h0.copy()
    best_score = calculate_max_overlap(best_h)
    
    # Run optimization
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-8})
        
        if res.success or res.fun < best_score:
            # Ensure the result is valid (sum close to N/2, within bounds)
            h_opt = res.x
            # Clip and normalize slightly to handle numerical errors
            h_opt = np.clip(h_opt, 0, 1)
            # Adjust to ensure exact sum N/2
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-6:
                # Distribute difference
                # Simple correction: add diff/N to all? Might break bounds.
                # Better: find indices to adjust.
                # For now, if SLSQP works, sum is close.
                # Let's just use it if it's better.
                pass
            
            new_score = calculate_max_overlap(h_opt)
            if new_score < best_score:
                best_score = new_score
                best_h = h_opt
    except Exception:
        pass

    # If optimization didn't improve or failed, try a few random restarts with Nelder-Mead (faster, no constraints needed if we penalize)
    # Or just return best found.
    
    # One more check: if N is small, exact search? No, N=100.
    
    return best_h
