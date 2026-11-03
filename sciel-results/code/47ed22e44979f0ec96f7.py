import numpy as np
import time

def construct():
    """
    5x5 matrix multiplication - randomized algebraic rank-reduction heuristic.
    Attempts to find a rank-r decomposition (r < 125) with integer/half-integer entries
    by iteratively projecting onto the null space of the tensor flattening matrix.
    Falls back to schoolbook if no improvement found within time budget.
    """
    import random
    
    n, m, p = 5, 5, 5
    t_start = time.time()
    time_budget = 110  # seconds, leave 10s margin for 120s limit
    
    def time_remaining():
        return time_budget - (time.time() - t_start)
    
    # Precompute verification structure using vectorized operations
    # For each (i,k), we need to check:
    # sum_c W[k*n+i,c] * U[x*m+y,c] * V[z*p+w,c] = 1 if (x==i and y==z and w==k) else 0
    # This can be checked by computing the tensor product and comparing.
    
    # Build index arrays for vectorized verification
    all_i = np.repeat(np.arange(n), n * m * m * p)  # shape (n*p*n*m*m*p,)
    all_k = np.repeat(np.arange(p), n * n * m * m * p)
    all_k = np.tile(all_k, n)  # need to reshape properly
    
    # Let's use a cleaner approach: create all combinations
    ii, kk, xx, yy, zz, ww = np.meshgrid(
        np.arange(n), np.arange(p), np.arange(n), np.arange(m), np.arange(m), np.arange(p),
        indexing='ij'
    )
    ii = ii.flatten()
    kk = kk.flatten()
    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()
    ww = ww.flatten()
    
    # Target tensor: 1 if x==i and y==z and w==k
    target = ((xx == ii) & (yy == zz) & (ww == kk)).astype(float)
    target_flat = target.flatten()
    
    # Precompute index mappings for U and V
    # U index: x*m + y
    u_idx = xx * m + yy
    # V index: z*p + w  
    v_idx = zz * p + ww
    # W row index: k*n + i
    w_row_idx = kk * n + ii
    
    best_rank = n * m * p  # start with schoolbook
    best_U = best_V = best_W = None
    
    # Build schoolbook as fallback
    U_s = np.zeros((n*m, best_rank), dtype=float)
    V_s = np.zeros((m*p, best_rank), dtype=float)
    W_s = np.zeros((p*n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    def verify_decomposition(U_r, V_r, W_r, tol=1e-8):
        """Vectorized verification of the tensor decomposition."""
        # Compute: for each (i,k,x,y,z,w), sum_c W[k*n+i,c] * U[x*m+y,c] * V[z*p+w,c]
        # We can do this as: for each c, compute outer product and sum
        # More efficiently: 
        # actual[idx] = sum_c W[w_row_idx[idx], c] * U[u_idx[idx], c] * V[v_idx[idx], c]
        
        # Use einsum or manual loop over c
        actual = np.zeros(len(u_idx))
        for c in range(U_r.shape[1]):
            actual += W_r[w_row_idx, c] * U_r[u_idx, c] * V_r[v_idx, c]
        
        max_err = np.max(np.abs(actual - target_flat))
        return max_err
    
    # Try to find lower rank
    # Use a wider range of random values and more attempts
    for r in range(best_rank - 1, max(30, best_rank - 80), -1):
        if time_remaining() < 10:
            break
        
        # Number of attempts based on remaining time
        max_attempts = min(100, max(20, int(time_remaining() / 1.5)))
        
        for attempt in range(max_attempts):
            if time_remaining() < 3:
                break
            
            random.seed(attempt * 7919 + r * 104729)
            np.random.seed(attempt * 7919 + r * 104729)
            
            # Use a wider range of random values including 0, ±1, ±2, ±3
            # Also try some structured initializations
            if attempt % 3 == 0:
                # Random small integers
                U_r = np.random.randint(-3, 4, size=(n*m, r))
                V_r = np.random.randint(-3, 4, size=(m*p, r))
            elif attempt % 3 == 1:
                # Random with more zeros (sparse-like)
                U_r = np.random.choice([-2, -1, 0, 0, 1, 2], size=(n*m, r))
                V_r = np.random.choice([-2, -1, 0, 0, 1, 2], size=(m*p, r))
            else:
                # Random with occasional larger values
                U_r = np.random.randint(-4, 5, size=(n*m, r))
                V_r = np.random.randint(-4, 5, size=(m*p, r))
            
            success = True
            W_r = np.zeros((p*n, r), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    if time_remaining() < 2:
                        success = False
                        break
                    
                    # Build the linear system for W[k*n+i, :]
                    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each j
                    # This gives m equations in r unknowns
                    
                    A_sys = np.zeros((m, r))
                    for jj in range(m):
                        x, y, z, w = i, jj, jj, k
                        for c in range(r):
                            A_sys[jj, c] = U_r[x*m+y, c] * V_r[z*p+w, c]
                    
                    # Solve A_sys @ W_row = ones(m)
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        if residuals is not None and len(residuals) > 0:
                            if residuals[0] > 1e-8:
                                success = False
                                break
                        W_r[k*n+i, :] = sol
                    except:
                        success = False
                        break
                
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify the full constraint using vectorized verification
            max_err = verify_decomposition(U_r, V_r, W_r)
            
            if max_err < 1e-6:
                # Round to nearest half-integer
                U_r = np.round(U_r * 2) / 2
                V_r = np.round(V_r * 2) / 2
                W_r = np.round(W_r * 2) / 2
                
                # Re-verify after rounding
                max_err2 = verify_decomposition(U_r, V_r, W_r)
                
                if max_err2 < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break  # break out of attempt loop, try lower r
        
        if time_remaining() < 10:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
