import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values
    # K=2000 is a good balance. K=3000 might be possible if fast enough.
    K_values = [1000, 2000, 3000]
    
    for K in K_values:
        if time.time() > deadline - 30:  # Leave 30s for final processing
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        # Objective: maximize -sum f(k) log(k)/k
        # linprog minimizes, so c = sum f(k) log(k)/k
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        rows = []
        cols = []
        data = []
        
        # Construct sparse A_ub
        # Constraint: sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1
        chunk_size = 2000
        for x_start in range(1, max_x + 1, chunk_size):
            if time.time() > deadline - 30:
                break
            x_end = min(x_start + chunk_size, max_x + 1)
            x_chunk = np.arange(x_start, x_end)
            
            # Calculate values for this chunk
            # A[x, k] = - (x % k) / k  -- Wait, floor(x/k) - x/k = - (x % k) / k
            rems = x_chunk[:, None] % ks[None, :]
            vals = -rems / ks[None, :]
            
            # Extract non-zero elements
            # Note: (x % k) / k is 0 only if x % k == 0
            nz_rows, nz_cols = np.nonzero(vals)
            nz_vals = vals[nz_rows, nz_cols]
            
            # Map to global indices
            # x_chunk indices are 0-based in chunk, need to map to global x index
            # The constraint rows are indexed by x.
            # In A_ub, row i corresponds to x = i+1?
            # Let's just use x as the row index directly if we build full matrix?
            # No, we are building sparse.
            # The rows in A_ub correspond to the constraints.
            # We are adding constraints for x in [x_start, x_end].
            # These are rows x_start-1 to x_end-1 in the full constraint matrix?
            # Yes, if we assume 1-to-1 mapping of x to constraint index.
            
            global_rows = x_chunk[nz_rows] - 1
            global_cols = nz_cols
            
            rows.extend(global_rows)
            cols.extend(nz_cols)
            data.extend(nz_vals)
            
        # Add constraints for S = sum(g/k)
        # f(1) = -S must be in [-10, 10] => S in [-10, 10]
        # S = sum_{k=2}^K f(k)/k
        row_pos = 1.0 / ks
        row_neg = -1.0 / ks
        
        # Row max_x: S <= 10
        rows.extend([max_x] * n_vars)
        cols.extend(np.arange(n_vars))
        data.extend(row_pos)
        
        # Row max_x+1: -S <= 10
        rows.extend([max_x + 1] * n_vars)
        cols.extend(np.arange(n_vars))
        data.extend(row_neg)
        
        A_ub_full = coo_matrix((data, (rows, cols)), shape=(max_x + 2, n_vars)).tocsr()
        b_ub_full = np.concatenate([np.ones(max_x), [10.0, 10.0]])
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            # Use highs method
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
            
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
                        
        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to a simple valid solution
        # Using the small known-good solution from inspiration
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
