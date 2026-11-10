import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 630.0
    
    # Parameters
    K = 200  # Maximum key in domain
    
    # Variables: f(2) ... f(K)
    # Index 0 corresponds to k=2, Index 1 to k=3, ..., Index K-2 to k=K
    n_vars = K - 1
    ks = np.arange(2, K + 1)
    
    # Objective: Minimize sum( (log(k)/k) * f(k) )
    # c_k = log(k) / k
    c = np.log(ks) / ks
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Constraints:
    # 1. M(x) <= 1 for x in [1, 10K]
    # M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1
    # Let A[i, j] = floor(x_i / k_j) - x_i / k_j
    # We want A @ f <= 1
    
    x_vals = np.arange(1, 10 * K + 1)
    n_constraints = len(x_vals)
    
    # Build A matrix
    # A shape: (10K, K-1)
    # This might be large but 2000 x 199 is small.
    A = np.zeros((n_constraints, n_vars))
    for i, x in enumerate(x_vals):
        for j, k in enumerate(ks):
            A[i, j] = np.floor(x / k) - x / k
            
    b = np.ones(n_constraints)
    
    # 2. Constraint on f(1) magnitude to avoid clipping issues
    # f(1) = - sum(f(k)/k)
    # -10 <= -sum(f(k)/k) <= 10
    # -10 <= sum(f(k)/k) <= 10
    # sum(f(k)/k) <= 10  =>  (1/k) @ f <= 10
    # sum(f(k)/k) >= -10 => - (1/k) @ f <= 10
    
    # Add these to A and b
    # We need to append rows to A
    A_new = np.vstack([A, (1.0 / ks).reshape(1, -1), -(1.0 / ks).reshape(1, -1)])
    b_new = np.append(b, [10.0, 10.0])
    
    # Solve LP
    # Use HiGHS method for speed and robustness
    try:
        res = opt.linprog(c, A_ub=A_new, b_ub=b_new, bounds=bounds, method='highs')
        
        if res.status == 0:
            f_vals = res.x
            # Construct full domain and values
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            
            # f(1)
            sum_fk_over_k = np.sum(f_vals / ks)
            f1 = -sum_fk_over_k
            values[0] = f1
            values[1:] = f_vals
            
            # Check time
            if time.time() > deadline - 10:
                # Fallback to simple Mobius if too close to deadline
                mu = np.zeros(K)
                for i in range(1, K):
                    mu[i] = 1
                # Simple Mobius calculation (slow but safe fallback)
                # Actually, just return the seed if we are in trouble
                pass
            
            return domain, values
        else:
            # Fallback to seed if LP fails
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    except Exception:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
