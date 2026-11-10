# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 600s limit, keep a safety margin

    # We want to maximize a = -sum_k f(k) * log(k) / k
    # subject to:
    # 1. M(x) = sum_k f(k) * floor(x/k) <= 1 for all x in [1, 10K]
    # 2. sum_k f(k) / k = 0 (after adjustment)
    # 3. f(k) in [-10, 10] (clipped, but we can constrain to this range)
    
    # The evaluator computes:
    # S = sum_k f(k) / k
    # f(1) = f(1) - S
    # Then M(x) = sum_k f(k) * floor(x/k) <= 1.0001
    # Metric a = -sum_k f(k) * log(k) / k
    
    # Note: after adjustment, sum_k f(k)/k = 0.
    # Let g(k) = f(k) for k > 1, and f(1) = -sum_{k>1} g(k)/k.
    # Then M(x) = f(1)*floor(x/1) + sum_{k>1} g(k)*floor(x/k)
    #           = -sum_{k>1} g(k)/k + sum_{k>1} g(k)*floor(x/k)
    #           = sum_{k>1} g(k) * (floor(x/k) - 1/k)
    
    # We want to maximize a = -f(1)*log(1)/1 - sum_{k>1} g(k)*log(k)/k
    # Since log(1) = 0, a = -sum_{k>1} g(k)*log(k)/k
    
    # So the problem becomes:
    # Maximize -sum_{k=2}^K g(k)*log(k)/k
    # Subject to:
    #   sum_{k=2}^K g(k) * (floor(x/k) - 1/k) <= 1 for all x in [1, 10K]
    #   -10 <= g(k) <= 10 for k = 2..K
    #   f(1) = -sum_{k=2}^K g(k)/k must be in [-10, 10]
    
    # We will solve the LP:
    # Maximize -sum c_k g_k => minimize sum c_k g_k
    # Subject to: A @ g <= b
    # where A_{x, k} = floor(x/k) - 1/k
    
    # We'll try multiple K values and pick the best result
    
    best_score = -np.inf
    best_result = None
    
    from scipy.optimize import linprog
    
    # Try different K values
    K_values = [200, 500, 1000, 1500, 2000]
    
    for K in K_values:
        # Check time
        if time.time() > deadline - 60:
            break
            
        n_vars = K - 1  # g_2, ..., g_K
        if n_vars <= 0:
            continue
        
        # Objective: maximize -sum c_k g_k => minimize sum c_k g_k
        c = np.zeros(n_vars)
        for i, k in enumerate(range(2, K+1)):
            c[i] = math.log(k) / k
        
        # Constraints: A @ g <= b
        # We need to select x values. Let's use x = 1, 2, ..., 10K.
        x_vals = np.arange(1, 10*K + 1)
        n_constraints = len(x_vals)
        
        # k_vals = 2, 3, ..., K
        k_vals = np.arange(2, K+1)
        
        # Build A matrix: A[i, j] = floor(x_i / k_j) - 1/k_j
        # This is a 10K x (K-1) matrix
        # For K=2000, this is 20000 x 1999, which is ~320 MB, might be too large
        # Let's be more selective with constraints
        
        # Instead of all x, use critical points where floor(x/k) changes
        # The maximum of M(x) is likely at x = m*k for some m, k
        # Let's use x = 1, 2, ..., 10K but subsample if too large
        
        if n_constraints * n_vars > 10_000_000:
            # Subsample constraints
            step = max(1, n_constraints // 5000)
            x_vals = x_vals[::step]
            n_constraints = len(x_vals)
        
        A = np.floor(x_vals[:, None] / k_vals[None, :]) - 1.0 / k_vals[None, :]
        b = np.ones(n_constraints)
        
        # Bounds for g_k: [-10, 10]
        bounds = [(-10, 10)] * n_vars
        
        try:
            # Use HiGHS solver which is fast
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', 
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-9})
            
            if res.success:
                g = res.x
                # Compute f(1)
                f1 = -np.sum(g / k_vals)
                
                # Check if f1 is within [-10, 10]
                if abs(f1) > 10:
                    # Clipping would occur, so this might not be valid
                    # But let's check the actual M(x) values
                    pass
                
                # Construct domain and values
                domain = np.arange(1, K+1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[0] = f1
                values[1:] = g
                
                # Verify constraints: check M(x) for all x in [1, 10K]
                # M(x) = f1 * floor(x/1) + sum_{k=2}^K g_k * floor(x/k)
                # = f1 * x + sum_{k=2}^K g_k * floor(x/k)
                
                # Check a subset of x values for efficiency
                check_x = np.arange(1, 10*K + 1)
                if len(check_x) > 10000:
                    check_x = check_x[::max(1, len(check_x)//10000)]
                
                # Compute M(x) for all check_x
                # M(x) = f1 * x + sum_k g_k * floor(x/k)
                # We can compute this efficiently using vectorization
                
                max_M = -np.inf
                for x in check_x:
                    M_x = f1 * x + np.sum(g * np.floor(x / k_vals))
                    if M_x > max_M:
                        max_M = M_x
                
                if max_M <= 1.0001:
                    # Compute the metric
                    a = -np.sum(g * np.log(k_vals) / k_vals)
                    # Note: f(1) contributes 0 to a since log(1) = 0
                    
                    if a > best_score:
                        best_score = a
                        best_result = (domain, values)
        except Exception as e:
            # LP failed, continue to next K
            continue
    
    # If we found a valid result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: Use a simple heuristic
    # Try f(k) = -1 for k = 2, 3, 5, 30 and adjust f(1)
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
