import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 550  # Leave 80s margin for safety
    
    # Try increasing K - the LP solver (HiGHS) should handle larger problems
    # K=1000 gave 1.0745. Try K=2000 or K=3000.
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0
    
    for K in [2000, 3000, 5000]:
        if time.time() - start_time > deadline - 30:
            break
        
        t0 = time.time()
        
        # Build constraint matrix B: B[x-1, j-1] = floor(x/j) - x/j
        x_vals = np.arange(1, 10 * K + 1)[:, np.newaxis]  # (10K, 1)
        j_vals = np.arange(1, K + 1)[np.newaxis, :]      # (1, K)
        
        # Compute B efficiently
        # floor(x/j) - x/j = -frac(x/j)
        B = np.floor(x_vals / j_vals) - (x_vals / j_vals)
        
        t_build = time.time() - t0
        if t_build > deadline - 60:
            continue
        
        # Objective: minimize sum_{k=2}^K (log(k)/k) * f_k
        c = np.zeros(K)
        k_vals = np.arange(1, K + 1, dtype=np.float64)
        c[1:] = np.log(k_vals[1:]) / k_vals[1:]
        
        bounds = [(-10.0, 10.0)] * K
        
        t1 = time.time()
        res = opt.linprog(c, A_ub=B, b_ub=np.ones(10 * K), bounds=bounds, method='highs')
        t_solve = time.time() - t1
        
        if time.time() - start_time > deadline:
            break
        
        if res.success:
            f_vals = res.x
            
            # Compute the score to compare
            # a = -sum_k f(k) * log(k) / k
            a = -np.sum(f_vals * np.log(np.arange(1, K + 1, dtype=np.float64)) / np.arange(1, K + 1, dtype=np.float64))
            # Note: log(1)/1 = 0, so k=1 doesn't contribute
            a = -np.sum(f_vals[1:] * np.log(k_vals[1:]) / k_vals[1:])
            
            if a > best_score:
                best_score = a
                best_domain = np.arange(1, K + 1, dtype=np.int64)
                best_values = f_vals.astype(np.float64)
        
        del B, x_vals, j_vals
        if time.time() - start_time > deadline - 30:
            break
    
    return best_domain, best_values
