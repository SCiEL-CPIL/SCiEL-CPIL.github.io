import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 150.0 # Safety margin, evaluator kills at 150s

def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return np.ones(N)
        
    # Objective function
    # We fix sum(h) = N by parameterizing N-1 variables.
    # h = [x_0, ..., x_{N-2}, N - sum(x_0...x_{N-2})]
    
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return 1e10 # Penalize if running out of time
        
        h = np.empty(N)
        h[:N-1] = x
        h[N-1] = N - np.sum(x)
        
        c = np.convolve(h, h)
        S = N # Fixed
        
        # Soft max with beta
        # beta needs to be tuned. Start moderate.
        beta = 10.0
        # To avoid overflow in exp, shift by max
        max_c = np.max(c)
        # If max_c is very large, beta * max_c might overflow.
        # But h is bounded if we use L-BFGS-B with bounds?
        # We don't have bounds on x, so h can be large.
        # Let's add a penalty for large values.
        
        # Soft max
        # log(sum(exp(beta * c))) / beta
        # = max_c + log(sum(exp(beta * (c - max_c)))) / beta
        
        beta_c = beta * (c - max_c)
        # Clip to avoid underflow/overflow issues in exp if necessary, 
        # but c - max_c <= 0, so exp is <= 1. Safe.
        log_sum_exp = np.log(np.sum(np.exp(beta_c)))
        soft_max = max_c + log_sum_exp / beta
        
        val = 2.0 * N * soft_max / (S * S)
        
        # Regularization to prevent h from exploding
        reg = 0.01 * np.sum(h**2)
        
        return val + reg

    # Initial guess
    # Try a chirp-like sequence
    i = np.arange(N)
    # Real chirp
    h_init = np.cos(2 * np.pi * i**2 / (2 * N))
    # Scale to sum N
    h_init = h_init / np.sum(h_init) * N
    
    # Extract free variables
    x0 = h_init[:N-1]
    
    # Bounds for L-BFGS-B
    # Keep heights within reasonable range to prevent numerical issues
    # e.g., [-10, 10]
    bounds = [(-10.0, 10.0) for _ in range(N-1)]
    
    best_h = h_init
    best_val = np.inf
    
    # Annealing loop
    # We can run multiple optimizations with increasing beta
    betas = [1.0, 5.0, 10.0, 20.0]
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Redefine objective with current beta
        def obj_current(x):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return 1e10
            h = np.empty(N)
            h[:N-1] = x
            h[N-1] = N - np.sum(x)
            c = np.convolve(h, h)
            max_c = np.max(c)
            beta_c = beta * (c - max_c)
            log_sum_exp = np.log(np.sum(np.exp(beta_c)))
            soft_max = max_c + log_sum_exp / beta
            val = 2.0 * N * soft_max / (N * N)
            reg = 0.01 * np.sum(h**2)
            return val + reg

        # Optimize
        try:
            res = minimize(obj_current, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'maxfun': 1000})
            if res.success:
                h_new = np.empty(N)
                h_new[:N-1] = res.x
                h_new[N-1] = N - np.sum(res.x)
                
                # Calculate true value (hard max)
                c_new = np.convolve(h_new, h_new)
                true_val = abs(2.0 * N * np.max(c_new) / (N * N))
                
                if true_val < best_val:
                    best_val = true_val
                    best_h = h_new
                    x0 = res.x # Use this as starting point for next beta
        except Exception:
            pass
            
    # Final check
    if time.time() - START_TIME > TIME_LIMIT - 2:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
