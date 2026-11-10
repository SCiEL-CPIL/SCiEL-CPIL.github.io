import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    # Set a deadline guard
    start_time = time.time()
    deadline = 550  # Leave 50s margin for safety
    
    # Choose K - try larger K with sparse matrix
    # With sparse representation, we can afford larger K
    K = 2000
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to seed if something goes wrong immediately
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # Construct the constraint matrix B in sparse form
    # B[x-1, j-1] = floor(x/j) for x in 1..10K, j in 1..K
    # We want to minimize -c^T f subject to B f <= 1
    
    # Build sparse matrix
    rows = []
    cols = []
    data = []
    
    x_max = 10 * K
    for x in range(1, x_max + 1):
        # For this x, j ranges from 1 to min(x, K)
        j_max = min(x, K)
        if j_max > 0:
            js = np.arange(1, j_max + 1)
            vals = np.floor(x / js).astype(np.float64)
            rows.extend([x - 1] * len(js))
            cols.extend((js - 1).tolist())
            data.extend(vals.tolist())
    
    # Check time
    if time.time() - start_time > deadline:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    B_sparse = sp.csr_matrix((data, (rows, cols)), shape=(x_max, K))
    
    # Objective: Maximize sum_{k=2}^K (-log(k)/k) * f_k
    # So minimize sum_{k=2}^K (log(k)/k) * f_k
    c = np.zeros(K)
    k_vals = np.arange(1, K + 1)
    c[1:] = np.log(k_vals[1:]) / k_vals[1:]
    
    # Bounds: -10 <= f <= 10
    bounds = [(-10.0, 10.0)] * K
    
    # Solve LP using HiGHS with sparse matrix
    res = opt.linprog(c, A_ub=B_sparse, b_ub=np.ones(x_max), bounds=bounds, method='highs',
                      options={'presolve': True, 'mip_rel_gap': 0.0})
    
    # Check time
    if time.time() - start_time > deadline:
        if res.success:
            f_vals = res.x
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f_vals.astype(np.float64)
            return domain, values
        else:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    if res.success:
        f_vals = res.x
    else:
        # If LP fails, try a smaller K
        K_small = 500
        if time.time() - start_time > deadline:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
        
        x_max_small = 10 * K_small
        rows_s = []
        cols_s = []
        data_s = []
        for x in range(1, x_max_small + 1):
            j_max_s = min(x, K_small)
            if j_max_s > 0:
                js_s = np.arange(1, j_max_s + 1)
                vals_s = np.floor(x / js_s).astype(np.float64)
                rows_s.extend([x - 1] * len(js_s))
                cols_s.extend((js_s - 1).tolist())
                data_s.extend(vals_s.tolist())
        
        B_sparse_s = sp.csr_matrix((data_s, (rows_s, cols_s)), shape=(x_max_small, K_small))
        c_s = np.zeros(K_small)
        k_vals_s = np.arange(1, K_small + 1)
        c_s[1:] = np.log(k_vals_s[1:]) / k_vals_s[1:]
        bounds_s = [(-10.0, 10.0)] * K_small
        
        res_s = opt.linprog(c_s, A_ub=B_sparse_s, b_ub=np.ones(x_max_small), bounds=bounds_s, method='highs')
        
        if res_s.success:
            f_vals = res_s.x
            K = K_small
        else:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f_vals.astype(np.float64)
    
    return domain, values
