import numpy as np
from scipy.optimize import linprog
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 630  # evaluator kills at 630s, leave margin
    
    best_result = None
    best_score = -np.inf
    
    # Try increasing K values
    K_candidates = [100, 200, 400, 600, 800, 1000, 1500, 2000]
    
    for K in K_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # leave 120s safety margin
            break
        
        n_vars = K - 1  # f(2), ..., f(K)
        
        # Objective: minimize sum f(k) log(k)/k  (to maximize negative)
        c = np.array([math.log(k) / k for k in range(2, K + 1)])
        
        # M(x) constraints: sum_{k=2}^K f(k) (floor(x/k) - x/k) <= 1.0001
        # for x in [1, 10K]
        x_vals = np.arange(1, 10 * K + 1)
        k_vals = np.arange(2, K + 1)
        
        # Vectorized: A_ub[i, j] = floor(x_i / k_j) - x_i / k_j
        x_col = x_vals[:, np.newaxis]   # (10K, 1)
        k_row = k_vals[np.newaxis, :]   # (1, K-1)
        
        A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10K, K-1)
        b_ub = np.full(10 * K, 1.0001)
        
        # Additional constraint: |sum_{k=2}^K f(k)/k| <= 10
        # This ensures f(1) = -sum f(k)/k stays in [-10, 10]
        s_coeff = np.array([1.0 / k for k in range(2, K + 1)])
        
        # Add two rows: sum f(k)/k <= 10 and -sum f(k)/k <= 10
        A_ub = np.vstack([A_ub, s_coeff[np.newaxis, :], -s_coeff[np.newaxis, :]])
        b_ub = np.append(b_ub, [10.0, 10.0])
        
        # Bounds: f(k) in [-10, 10]
        bounds = [(-10, 10)] * n_vars
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # Compute the metric a = -sum f(k) log(k)/k
                obj_val = -np.sum(f_vals * c)
                
                # f(1) = -sum_{k=2}^K f(k)/k
                f1 = -np.sum(f_vals * s_coeff)
                
                if obj_val > best_score:
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[0] = f1  # f(1)
                    values[1:] = f_vals
                    best_result = (domain, values)
                    best_score = obj_val
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple mu-based approach
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
