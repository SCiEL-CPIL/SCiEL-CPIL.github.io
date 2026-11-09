import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_halfint_list(arr):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2.0
            if abs(rounded - int(rounded)) < 1e-9:
                row_list.append(int(rounded))
            else:
                row_list.append(rounded)
        result.append(row_list)
    return result

def _compute_residuals(params, n, m, p, r):
    """Compute the residual vector for least_squares."""
    # params is a flat array of size 71*r
    # U: n*m*r, V: m*p*r, W: p*n*r
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape((n * m, r))
    V = V_flat.reshape((m * p, r))
    W = W_flat.reshape((p * n, r))
    
    # Compute R_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
    # We can use einsum for this
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # We need to sum over c.
    
    # Reshape U to (n, m, r)
    U_3d = U.reshape((n, m, r))
    # Reshape V to (m, p, r)
    V_3d = V.reshape((m, p, r))
    # Reshape W to (p, n, r)
    W_3d = W.reshape((p, n, r))
    
    # Sum over c:
    # Term for (i,j,k) is sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    
    # This is a bit tricky with einsum directly because indices are mixed.
    # Let's do it in a loop over i, j, k but vectorize over c?
    # Or use tensordot?
    
    # Alternative:
    # For each c, we have a rank-1 tensor T_c = U_c \otimes V_c \otimes W_c (with specific index permutations)
    # U_c is (n, m), V_c is (m, p), W_c is (p, n)
    # We want sum_c U_c[i,j] V_c[j,k] W_c[k,i]
    
    # Let's just loop over i, j, k and use dot products.
    # n=3, m=5, p=7. 105 iterations.
    # Inside, dot product of length r.
    # This is fast enough for r up to 100.
    
    residuals = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            u_row = U_3d[i, j, :] # shape (r,)
            for k in range(p):
                v_row = V_3d[j, k, :] # shape (r,)
                w_row = W_3d[k, i, :] # shape (r,)
                # sum_c u_row[c] * v_row[c] * w_row[c]
                val = np.dot(u_row * v_row, w_row)
                residuals[i, j, k] = val - 1.0
                
    return residuals.flatten()

def _try_rank_scipy(r, n, m, p, max_seeds=10, time_budget=10.0):
    """Try to find a rank-r decomposition using scipy.optimize.least_squares."""
    start_time = time.time()
    
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    total_size = size_U + size_V + size_W
    
    for seed in range(max_seeds):
        if time.time() - start_time > time_budget:
            break
            
        np.random.seed(seed)
        
        # Initialization: Random small values
        # Scaling is important. If values are too large, gradient explodes.
        # If too small, it stays near 0.
        # Target sum is 1.
        # Let's try uniform distribution in [-1, 1]
        params0 = np.random.uniform(-1, 1, total_size)
        
        try:
            result = opt.least_squares(
                _compute_residuals, 
                params0, 
                args=(n, m, p, r),
                method='lm', # Levenberg-Marquardt, good for small-medium problems
                max_nfev=1000
            )
            
            if result.success or result.cost < 1e-10:
                # Extract U, V, W
                U_flat = result.x[:size_U]
                V_flat = result.x[size_U:size_U + size_V]
                W_flat = result.x[size_U + size_V:]
                
                U = U_flat.reshape((n * m, r))
                V = V_flat.reshape((m * p, r))
                W = W_flat.reshape((p * n, r))
                
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded, V_rounded, W_rounded
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Theoretical lower bound
    r_min = max(n*m, m*p, n*p)
    
    # Try ranks from r_min up to r_school - 1
    # We want the lowest rank, so we start from r_min and go up.
    # If we find one, we return it immediately? 
    # The prompt says "Find a rank-minimal exact tensor decomposition".
    # So we should try to find the absolute minimum.
    # However, if we can't find the minimum in time, we should return the best found.
    
    # Strategy:
    # 1. Try r_min. If found, return.
    # 2. Try r_min + 1. If found, return.
    # ...
    # This might be slow if r_min is hard.
    # Alternative: Try a few ranks in parallel? No, sequential is fine.
    
    # Let's try to find ANY rank < 105 first, then refine?
    # No, "rank-minimal" implies we should try to get as low as possible.
    # But if we don't know if 35 is possible, we might waste time.
    # Let's try 35, 36, ... up to 104.
    
    # To be safe with time, we will limit the number of seeds per rank.
    # If we fail to find a solution for a rank, we move to the next.
    
    for r in range(r_min, rank_school):
        if time.time() > deadline - 5:
            break
            
        # Allocate more time for lower ranks as they are more likely to be the answer
        # and harder to find.
        time_per_rank = 5.0 if r < 50 else 2.0
        
        result = _try_rank_scipy(r, n, m, p, max_seeds=5, time_budget=time_per_rank)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": _to_halfint_list(U),
                "V": _to_halfint_list(V),
                "W": _to_halfint_list(W)
            }
            # Since we are iterating from low to high, the first one we find is the minimum
            # among those we tried. We should return immediately?
            # "Find a rank-minimal...". Yes, if we assume we hit the minimum.
            # But what if 35 is impossible and 36 is possible? We find 36.
            # What if 35 is possible but we failed to find it?
            # Given the time constraint, we can't guarantee we found the absolute minimum.
            # But returning the first one found (lowest r tried) is a good heuristic.
            # Let's return immediately to save time?
            # Or continue to see if we can find a lower one? We are already going low to high.
            # So the first one found IS the lowest rank we found.
            # We can break and return.
            return best_result
            
    return best_result
