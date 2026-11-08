def _try_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-r decomposition using Alternating Least Squares (ALS).
    Returns (U, V, W) if successful, else None.
    """
    if rank < max(n*m, m*p, p*n):
        # Theoretical lower bound is higher, but let's be safe
        # Actually, rank must be at least max(n*m, m*p, p*n) is NOT true.
        # The rank of the tensor can be lower.
        # But if rank < max(n*m, m*p, p*n), the matrices U, V, W might not have full row rank.
        # ALS still works.
        pass

    np.random.seed(rank) # Deterministic based on rank
    
    # Initialize U, V, W
    # Random initialization
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize to prevent overflow/underflow
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    best_residual = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    max_iters = 100
    tol = 1e-8
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                # Build A: p x rank
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # We can compute this efficiently
                # V_j = V[j*p : (j+1)*p]  (p x rank)
                # W_i = W[i : p*n : n]    (p x rank) -- careful with indexing
                # W is (p*n, rank). Rows are indexed by k*n + i.
                # So for fixed i, the rows are W[0*n+i], W[1*n+i], ..., W[(p-1)*n+i]
                W_i = W[np.arange(p) * n + i, :] # (p, rank)
                V_j = V[j * p : (j + 1) * p, :]  # (p, rank)
                A = V_j * W_i # (p, rank)
                
                b = np.ones(p)
                try:
                    # Solve A x = b
                    # Use lstsq
                    u_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[idx_u, :] = u_new
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                # Build B: n x rank
                # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                U_j = U[:, j] # (n, rank) -- careful: U is (n*m, rank). Rows i*m+j.
                # U[i, j] corresponds to row i*m + j.
                # So for fixed j, we need U[0*m+j], U[1*m+j], ...
                U_j = U[np.arange(n) * m + j, :] # (n, rank)
                W_k = W[k * n : (k + 1) * n, :]  # (n, rank) -- Rows k*n + i for i=0..n-1
                B = U_j * W_k # (n, rank)
                
                b = np.ones(n)
                try:
                    v_new, residuals, rank_b, sv = np.linalg.lstsq(B, b, rcond=None)
                    V[idx_v, :] = v_new
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                # Build C: m x rank
                # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                U_i = U[i * m : (i + 1) * m, :] # (m, rank)
                V_k = V[np.arange(m) * p + k, :] # (m, rank) -- Rows j*p + k for j=0..m-1
                C = U_i * V_k # (m, rank)
                
                b = np.ones(m)
                try:
                    w_new, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
                    W[idx_w, :] = w_new
                except:
                    pass

        # Check convergence
        # Compute residual
        # T_ijk = sum_r U_ij_r V_jk_r W_ki_r
        # We can check a few random samples or all samples
        # All samples: n*m*p = 144 checks.
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    residual += (val - 1.0) ** 2
        residual = np.sqrt(residual / (n * m * p))
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if residual < tol:
            return (best_U, best_V, best_W)
            
    return None
