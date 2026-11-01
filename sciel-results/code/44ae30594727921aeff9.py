# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _als_decompose(T, rank, n, m, p, max_iters=1000, tol=1e-12, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (n*m, m*p, p*n)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    # Normalize initial vectors to avoid scaling issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
            U[:, r] /= norm_u
            V[:, r] /= norm_v
            W[:, r] /= norm_w
            
    prev_error = float('inf')
    
    for it in range(max_iters):
        # Update U
        for a in range(dim_a):
            M = np.einsum('br,cr->bc r', V, W)
            M_flat = M.reshape(dim_b * dim_c, rank)
            T_flat = T[a, :, :].reshape(-1)
            u_a, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        for b in range(dim_b):
            M = np.einsum('ar,cr->ac r', U, W)
            M_flat = M.reshape(dim_a * dim_c, rank)
            T_flat = T[:, b, :].reshape(-1)
            v_b, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        for c in range(dim_c):
            M = np.einsum('ar,br->ab r', U, V)
            M_flat = M.reshape(dim_a * dim_b, rank)
            T_flat = T[:, :, c].reshape(-1)
            w_c, residuals, rank_ls, s = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-6:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-6:
        return U, V, W
    return None

def _try_rank23(n, m, p, deadline):
    """
    Attempt to find a rank-23 decomposition using ALS.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Get schoolbook decomposition
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    
    all_indices = list(range(total_terms))
    
    attempts = 0
    max_attempts = 2000 # Increased attempts
    
    random.seed(42)
    
    # Strategy 1: Drop 4 terms from schoolbook
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Pick 4 indices to drop
        drop_indices = random.sample(all_indices, 4)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        # Run ALS
        result = _als_decompose(T, 23, n, m, p, max_iters=300, tol=1e-12, 
                                init_U=init_U, init_V=init_V, init_W=init_W)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
                
    # Strategy 2: Random initialization
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            result = _als_decompose(T, 23, n, m, p, max_iters=500, tol=1e-12)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    return U, V, W
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try to find rank 23
    if time.time() < deadline:
        result = _try_rank23(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
