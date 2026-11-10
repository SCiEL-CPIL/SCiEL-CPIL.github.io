import numpy as np
from scipy.optimize import linprog
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 590  # seconds, safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Try increasing K values
    K_values = [100, 200, 400, 800, 1600]
    
    for K in K_values:
        if time.time() - start_time > deadline - 30:
            break
            
        n_vars = K - 1
        
        # Objective: maximize -sum f(k) log(k)/k
        # linprog minimizes c^T x, so c = log(k)/k
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        c = np.log(k_vals) / k_vals
        
        # Constraints: sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1 for x in [1, 10K]
        # Note: floor(x/k) - x/k is in [-1, 0)
        x_vals = np.arange(1, 10 * K + 1, dtype=np.float64)
        
        # Build A_ub: shape (10K, K-1)
        # A_ub[x_idx, k_idx] = floor(x/k) - x/k
        x_col = x_vals[:, np.newaxis]   # (10K, 1)
        k_row = k_vals[np.newaxis, :]   # (1, K-1)
        
        A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10K, K-1)
        b_ub = np.ones(10 * K)
        
        # Bounds
        bounds = [(-10, 10)] * n_vars
        
        # Solve LP with time limit
        t0 = time.time()
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            solve_time = time.time() - t0
            
            if res.success:
                f_vals = res.x
                obj_val = -np.sum(f_vals * c)
                
                if obj_val > best_score:
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[1:] = f_vals
                    best_result = (domain, values)
                    best_score = obj_val
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: use mu(k) scaled
        K = 100
        mu = np.zeros(K + 1, dtype=int)
        mu[1] = 1
        for i in range(2, K + 1):
            if mu[i] == 0:
                for j in range(i, K + 1, i):
                    mu[j] = -mu[j]
                    if j % (i * i) == 0:
                        mu[j] = 0
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = mu[2:K + 1].astype(np.float64)
        return domain, values
    
    return best_result
