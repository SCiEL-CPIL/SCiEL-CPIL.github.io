def _als_decompose_v2(n, m, p, rank, max_iter=100, tol=1e-10, time_budget=20):
    start = time.time()
    
    # Target tensor T[a,b,c]:
    # a = i'*m + j' (U index)
    # b = j'*p + k' (V index)
    # c = k'*n + i (W index)
    # T[a,b,c] = 1 if i'==i and k'==k (i.e., a//m == c%n and b%p == c//n), else 0
    
    # Precompute target
    # T has shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        ip = a // m
        jp = a % m
        for b in range(m*p):
            jb = b // p
            kp = b % p
            for c in range(p*n):
                kc = c // n
                i = c % n
                if ip == i and kp == kc:
                    T[a, b, c] = 1.0
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Update U: for each a, solve least squares
        for a in range(n*m):
            # Build A_a: shape (m*p * p*n, rank)
            # Row (b, c): [V[b,0]*W[c,0], ..., V[b,rank-1]*W[c,rank-1]]
            # Vectorize: A_a[b*n_p + c, r] = V[b,r] * W[c,r] where n_p = p*n
            A_a = (V[:, :, None] * W[None, :, :]).reshape(m*p, p*n, rank).reshape(m*p*p*n, rank)
            t_a = T[a].reshape(-1)
            U[a, :], _, _, _ = np.linalg.lstsq(A_a, t_a, rcond=None)
        
        # Update V: for each b, solve least squares
        for b in range(m*p):
            # Constraint: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # For fixed b: U[a,r] * W[c,r] * V[b,r] summed over r = T[a,b,c]
            # A_b[a*n_p + c, r] = U[a,r] * W[c,r]
            A_b = (U[:, :, None] * W[None, :, :]).reshape(n*m, p*n, rank).reshape(n*m*p*n, rank)
            t_b = T[:, b, :].reshape(-1)
            V[b, :], _, _, _ = np.linalg.lstsq(A_b, t_b, rcond=None)
        
        # Update W: for each c, solve least squares
        for c in range(p*n):
            # A_c[a*m_p + b, r] = U[a,r] * V[b,r]
            A_c = (U[:, :, None] * V[None, :, :]).reshape(n*m, m*p, rank).reshape(n*m*m*p, rank)
            t_c = T[:, :, c].reshape(-1)
            W[c, :], _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
        
        # Compute error
        # err = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
        # Vectorize: for each r, outer product
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            recon += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        err_val = np.sum((recon - T)**2)
        
        if err_val < best_err:
            best_err = err_val
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err_val < tol:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None
