# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0  # 600 seconds total budget
    
    # We will try increasing K.
    # Dense matrix size is 10K * (K-1).
    # K=2000 -> 40M floats ~ 320MB.
    # K=3000 -> 90M floats ~ 720MB.
    # K=4000 -> 160M floats ~ 1.2GB.
    # K=5000 -> 250M floats ~ 2GB.
    
    # Start with K=2000, try to go higher if time permits.
    K_candidates = [2000, 3000, 4000]
    
    best_score = 0.0
    best_domain = None
    best_values = None
    
    # Fallback to the seed solution if LP fails or times out
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    for K in K_candidates:
        if time.time() > deadline - 120: # 120s margin for final checks and larger K attempts
            break
            
        try:
            n_vars = K - 1
            ks = np.arange(2, K + 1)
            
            # Objective: maximize a = - sum_{k=2}^K f(k) log(k)/k
            # linprog minimizes, so c = log(k)/k
            c_obj = np.log(ks) / ks
            
            # Constraints:
            # 1. f(1) bounds: -10 <= -sum f(k)/k <= 10
            # sum f(k)/k <= 10
            # -sum f(k)/k <= 10
            v_vec = 1.0 / ks
            
            # 2. M(x) <= 1.0001
            # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
            # x in 1..10K
            
            num_x = 10 * K
            total_rows = 2 + num_x
            
            # Construct A_ub
            # Rows 0, 1: f(1) constraints
            # Rows 2..: M(x) constraints
            
            # Pre-allocate dense matrix if size is reasonable
            # Size: total_rows x n_vars
            # For K=3000, 30002 x 2999 ~ 9e7.
            
            A_ub = np.zeros((total_rows, n_vars), dtype=np.float64)
            
            # Row 0: v_vec
            A_ub[0, :] = v_vec
            # Row 1: -v_vec
            A_ub[1, :] = -v_vec
            
            # Rows 2..: M(x)
            # x from 1 to 10K
            x_vals = np.arange(1, num_x + 1)[:, np.newaxis] # (num_x, 1)
            k_vals = ks[np.newaxis, :] # (1, n_vars)
            
            # Calculate A_M = floor(x/k) - x/k
            # This is a large operation.
            # floor(x/k) is integer, x/k is float.
            A_M = np.floor(x_vals / k_vals) - x_vals / k_vals
            
            A_ub[2:, :] = A_M
            
            b_ub = np.full(total_rows, 1.0001)
            b_ub[0] = 10.0
            b_ub[1] = 10.0
            
            bounds = [(-10.0, 10.0) for _ in range(n_vars)]
            
            # Solve
            # Use highs method, which is robust for dense matrices too
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                continue
                
            f_rest = res.x
            f1 = -np.sum(f_rest / ks)
            
            # Construct full domain and values
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            values[0] = f1
            values[1:] = f_rest
            
            # Calculate score
            log_k = np.log(domain.astype(np.float64))
            a = -np.sum(values * log_k / domain.astype(np.float64))
            
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue
            
    if best_domain is not None:
        return best_domain, best_values
    else:
        return seed_domain, seed_values

# EVOLVE-BLOCK-END
