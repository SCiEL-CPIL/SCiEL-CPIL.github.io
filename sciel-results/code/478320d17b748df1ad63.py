def _try_find_decomposition(rank, n, m, p, time_limit, seed):
    import numpy as np
    import time
    
    start = time.time()
    rng = np.random.default_rng(seed)
    
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Alternating Least Squares
    # We want to minimize || T - sum_r u_r v_r^T w_r^T ||
    # But T is the multiplication tensor.
    # We can form the target tensor T of shape (n*m, m*p, n*p) ?
    # No, the tensor is (n*m, m*p, p*n) with trace convention.
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape T to 2D for ALS: (n*m * m*p, p*n) ?
    # The model is: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # This is not a simple matrix product.
    # It is a tensor decomposition.
    # ALS for tensor decomposition:
    # Fix V, W. Solve for U.
    # For each a, T[a, :, :] = sum_r U[a, r] (V[:, r] * W[:, r]) ?
    # No. T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # For fixed a, T[a, b, c] is a matrix in (b, c).
    # It is sum_r U[a, r] (v_r * w_r^T) where v_r = V[:, r], w_r = W[:, r].
    # So T[a] = sum_r U[a, r] M_r, where M_r = np.outer(V[:, r], W[:, r]).
    # This is a linear regression for U[a, :].
    # The design matrix is [M_1, M_2, ..., M_rank] flattened to (m*p*p*n, rank).
    
    # This is expensive but doable for small ranks.
    # n=2, m=5, p=5.
    # U: (10, R)
    # V: (25, R)
    # W: (10, R)
    # T: (10, 25, 10)
    
    # For each a (10 times):
    # Target: T[a] (25, 10) -> flatten to 250.
    # Design: For each r, outer(V[:, r], W[:, r]) flattened to 250.
    # Solve U[a, :] = lstsq(Design, Target).
    
    # Similarly for V and W.
    
    for iter in range(50):
        if time.time() - start > time_limit:
            break
            
        # Solve U
        Design_U = np.zeros((m * p * p * n, rank))
        for r in range(rank):
            M_r = np.outer(V[:, r], W[:, r])
            Design_U[:, r] = M_r.flatten()
        
        for a in range(n * m):
            Target_U = T[a, :, :].flatten()
            U[a, :], residuals, rank_solve, sv = np.linalg.lstsq(Design_U, Target_U, rcond=None)
            
        # Solve V
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed b: T[:, b, :] (n*m, p*n)
        # = sum_r V[b, r] (U[:, r] * W[:, r]) ?
        # T[a, b, c] = sum_r (U[a, r] W[c, r]) V[b, r].
        # Let X_r = np.outer(U[:, r], W[:, r]).
        # T[:, b, :] = sum_r V[b, r] X_r.
        # Design_V: (n*m*p*n, rank)
        Design_V = np.zeros((n * m * p * n, rank))
        for r in range(rank):
            X_r = np.outer(U[:, r], W[:, r])
            Design_V[:, r] = X_r.flatten()
            
        for b in range(m * p):
            Target_V = T[:, b, :].flatten()
            V[b, :], residuals, rank_solve, sv = np.linalg.lstsq(Design_V, Target_V, rcond=None)
            
        # Solve W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For fixed c: T[:, :, c] (n*m, m*p)
        # = sum_r W[c, r] (U[:, r] * V[:, r]) ?
        # T[a, b, c] = sum_r (U[a, r] V[b, r]) W[c, r].
        # Let Y_r = np.outer(U[:, r], V[:, r]).
        # T[:, :, c] = sum_r W[c, r] Y_r.
        # Design_W: (n*m*m*p, rank)
        Design_W = np.zeros((n * m * m * p, rank))
        for r in range(rank):
            Y_r = np.outer(U[:, r], V[:, r])
            Design_W[:, r] = Y_r.flatten()
            
        for c in range(p * n):
            Target_W = T[:, :, c].flatten()
            W[c, :], residuals, rank_solve, sv = np.linalg.lstsq(Design_W, Target_W, rcond=None)
            
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
            
    return None
