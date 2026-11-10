# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k, subject to
    sum_k f(k)/k = 0 and M(x) = sum_k f(k) floor(x/k) <= 1.0001 for x in [1, 10K].
    
    Uses Linear Programming to find the optimal f(k) for k=2..K, with f(1) determined
    by the normalization constraint.
    """
    start_time = time.time()
    # Leave 30s margin for safety
    deadline = start_time + 570.0

    # Fallback seed solution
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 0.0

    # Calculate seed score to have a baseline
    # Seed values after normalization: f(1)=1, others -1,1
    # a_seed = log(2)/2 + log(3)/3 + log(5)/5 - log(30)/30
    best_score = (np.log(2)/2 + np.log(3)/3 + np.log(5)/5 - np.log(30)/30)

    # Try increasing K
    # Start with smaller K to ensure we have a valid result quickly
    for K in [50, 100, 200, 500, 1000]:
        # Check time
        if time.time() > deadline - 120.0:
            break
        
        try:
            n_vars = K - 1
            if n_vars <= 0:
                continue
            
            # Objective: Minimize sum f(k) log(k)/k for k=2..K
            # which is equivalent to Maximize -sum f(k) log(k)/k
            c = np.zeros(n_vars)
            ks = np.arange(2, K + 1)
            c[:] = np.log(ks) / ks
            
            # Constraints: M(x) <= 1.0001
            # M(x) = f(1)*1 + sum_{k=2}^K f(k) floor(x/k)
            # f(1) = -sum_{j=2}^K f(j)/j
            # M(x) = -sum_{j=2}^K f(j)/j + sum_{k=2}^K f(k) floor(x/k)
            # M(x) = sum_{k=2}^K f(k) (floor(x/k) - 1/k)
            # We require M(x) <= 1.0001
            
            x_max = 10 * K
            A_ub = np.zeros((x_max, n_vars))
            b_ub = np.full(x_max, 1.0001)
            
            # Vectorized construction of A_ub
            x_col = np.arange(1, x_max + 1, dtype=np.float64).reshape(-1, 1)
            k_row = ks.astype(np.float64).reshape(1, -1)
            
            # floor(x/k) - 1/k
            A_ub[:] = np.floor(x_col / k_row) - (1.0 / k_row)
            
            # Bounds for f(k): [-10, 10]
            bounds = [(-10.0, 10.0)] * n_vars
            
            # Solve LP
            # Using HiGHS solver which is fast and robust
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # Calculate f(1)
                f1 = -np.sum(f_vals / ks)
                
                # Construct full arrays
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.empty(K, dtype=np.float64)
                values[0] = f1
                values[1:] = f_vals
                
                # Calculate score a
                # a = -sum f(k) log(k)/k
                # log(1) = 0, so first term is 0
                log_k_over_k = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
                a = -np.sum(values * log_k_over_k)
                
                # Check if this is better than the current best
                if a > best_score:
                    best_score = a
                    best_domain = domain
                    best_values = values
                    
        except Exception:
            # If LP fails, continue to next K or break
            pass

    return best_domain, best_values

# EVOLVE-BLOCK-END
