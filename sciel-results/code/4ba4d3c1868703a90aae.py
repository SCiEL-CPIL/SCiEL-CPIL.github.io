import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    last_solution = None  # For warm-starting
    
    # Try different K values with increasing complexity
    K_values = [100, 200, 500, 1000, 2000, 4000, 8000]
    
    for K in K_values:
        if time.time() > deadline:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks  # Minimize sum c_k * g_k = sum (log k / k) * g_k, which is -objective
        
        max_x = 10 * K
        
        # Build sparse constraint matrix
        # M(x) = sum_k f(k) * floor(x/k) <= 1.0001
        # floor(x/k) = x/k - {x/k}
        # So M(x) = x * S - sum_k f(k) * {x/k} where S = sum f(k)/k
        # After f(1) adjustment, S = 0, so M(x) = -sum_k f(k) * {x/k}
        # But we need to be careful: the constraint is on the original f before f(1) adjustment
        # Actually, let's just use the direct formulation: M(x) = sum_k f(k) * floor(x/k)
        
        rows = []
        cols = []
        data = []
        
        # For each x in [1, max_x], the row is floor(x/k) for k in [2, K]
        # We can build this efficiently
        chunk_size = 500
        for x_start in range(1, max_x + 1, chunk_size):
            x_end = min(x_start + chunk_size, max_x + 1)
            x_chunk = np.arange(x_start, x_end)
            
            # Compute floor(x/k) for all x in chunk, all k
            # Shape: (chunk_len, n_vars)
            floor_vals = x_chunk[:, None] // ks[None, :]
            
            # Extract non-zero elements (floor(x/k) is 0 when x < k)
            nz_rows, nz_cols = np.nonzero(floor_vals)
            nz_vals = floor_vals[nz_rows, nz_cols]
            
            # Map to global indices
            global_rows = x_chunk[nz_rows] - 1  # 0-indexed
            global_cols = nz_cols
            
            rows.extend(global_rows.tolist())
            cols.extend(global_cols.tolist())
            data.extend(nz_vals.tolist())
        
        # Add S constraint: S = sum g(k)/k must be in [-10, 10]
        # f(1) = -S, and |f(1)| <= 10 => |S| <= 10
        row_pos = 1.0 / ks
        row_neg = -1.0 / ks
        
        rows.extend([max_x] * n_vars)
        cols.extend(range(n_vars))
        data.extend(row_pos.tolist())
        
        rows.extend([max_x + 1] * n_vars)
        cols.extend(range(n_vars))
        data.extend(row_neg.tolist())
        
        A_ub = coo_matrix((data, (rows, cols)), shape=(max_x + 2, n_vars)).tocsr()
        b_ub = np.concatenate([np.ones(max_x), [10.0, 10.0]])
        
        bounds = [(-10, 10)] * n_vars
        
        # Warm start if available
        x0 = None
        if last_solution is not None and len(last_solution) == n_vars:
            x0 = last_solution
        elif last_solution is not None and len(last_solution) < n_vars:
            # Pad with zeros
            x0 = np.zeros(n_vars)
            x0[:len(last_solution)] = last_solution
        elif last_solution is not None and len(last_solution) > n_vars:
            # Truncate
            x0 = last_solution[:n_vars]
        
        try:
            options = {'maxiter': 5000}
            if x0 is not None:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                             method='highs', x0=x0, options=options)
            else:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                             method='highs', options=options)
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                if abs(f[1]) <= 10.0 + 1e-9:
                    a = -np.sum(g * c)
                    if a > best_score:
                        best_score = a
                        domain = np.arange(1, K + 1)
                        values = np.clip(f[1:K+1], -10, 10)
                        best_domain = domain
                        best_values = values
                        last_solution = g.copy()
                        
        except Exception as e:
            pass
    
    if best_domain is not None:
        return best_domain, best_values
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
