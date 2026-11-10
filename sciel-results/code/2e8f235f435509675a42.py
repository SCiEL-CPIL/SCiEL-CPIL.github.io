import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # We formulate the problem as an LP.
    # Variables: f(1), f(2), ..., f(K)
    # Objective: maximize a = -sum f(k) log(k) / k
    #           => minimize c = sum (log(k) / k) * f(k)
    # Constraints:
    # 1. Normalization: sum f(k) / k = 0
    # 2. Feasibility: M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K]
    # 3. Bounds: -10 <= f(k) <= 10 (due to clipping in evaluator)
    
    # Choose K. K=200 is a good balance between approximation quality and LP size.
    # The LP size is roughly (10K) constraints x K variables.
    # For K=200, that's 2000 constraints x 200 vars. This is very fast for HiGHS.
    K = 200
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct the constraint matrix A_ub for M(x) <= 1
    # M(x) = sum_{k=1}^K f(k) * floor(x/k)
    # We need this for x = 1, ..., 10*K
    
    max_x = 10 * K
    n_vars = K
    n_cons = max_x
    
    # A_ub is n_cons x n_vars
    # A_ub[i, j] = floor((i+1) / (j+1))
    # Using broadcasting for speed
    x_vals = np.arange(1, max_x + 1)[:, np.newaxis]
    k_vals = np.arange(1, K + 1)[np.newaxis, :]
    A_ub = (x_vals // k_vals).astype(np.float64)
    
    b_ub = np.ones(max_x)
    
    # Objective function c
    # Minimize sum c_i f_i
    # c_i = log(i+1) / (i+1)
    c = np.log(np.arange(1, K + 1)) / np.arange(1, K + 1)
    
    # Equality constraint: sum f(k) / k = 0
    A_eq = (1.0 / np.arange(1, K + 1)).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10.0, 10.0)] * K
    
    # Solve LP
    # Method 'highs' is the default and most robust for this size
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to seed if LP fails
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
            
        f_vals = res.x
        
        # Filter out very small values to reduce domain size (optional but good for cleanliness)
        # The evaluator handles any distinct positive integers.
        # However, keeping all K is fine as long as K <= 200000.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = f_vals.astype(np.float64)
        
        return domain, values
        
    except Exception:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
