def _als_solve(n, m, p, rank, max_iter=100, tol=1e-12, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(max_iter):
        # Solve for V: for each (j,k), A[j,k] = U[:, j] * W[:, k] elementwise, shape (n, rank)
        # A is (m, p, n, rank), b is (m, p, n)
        # Solve A[j,k] @ V[j*p+k] = b[j,k]
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_jk = (U[:, j] * W[:, k]).T  # shape (n, rank) -- wait, U is (n*m, rank)
                # U[i*m + j, :] for i in range(n)
                U_sub = U.reshape(n, m, rank)[:, j, :]  # (n, rank)
                W_sub = W.reshape(p, n, rank)[:, k, :]  # (n, rank)
                A_jk = U_sub * W_sub  # (n, rank)
                b_jk = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A_jk, b_jk, rcond=None)
                V_new[j, k] = sol
        
        # Solve for W: for each (k,i), A[k,i] = U[i, :] * V[:, k] elementwise, shape (m, rank)
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                U_sub = U.reshape(n, m, rank)[i, :, :]  # (m, rank)
                V_sub = V.reshape(m, p, rank)[:, k, :]  # (m, rank)
                A_ki = U_sub * V_sub  # (m, rank)
                b_ki = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A_ki, b_ki, rcond=None)
                W_new[k, i] = sol
        
        # Solve for U: for each (i,j), A[i,j] = V[j, :] * W[:, i] elementwise, shape (p, rank)
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                V_sub = V.reshape(m, p, rank)[j, :, :]  # (p, rank)
                W_sub = W.reshape(p, n, rank)[:, i, :]  # (p, rank)
                A_ij = V_sub * W_sub  # (p, rank)
                b_ij = np.ones(p)
                sol, _, _, _ = np.linalg.lstsq(A_ij, b_ij, rcond=None)
                U_new[i, j] = sol
        
        diff = (np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W) + np.linalg.norm(U_new - U))
        U, V, W = U_new, V_new, W_new
        
        if diff < tol:
            if _validate(U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist(), n, m, p):
                return U.reshape(n*m, rank), V.reshape(m*p, rank), W.reshape(p*n, rank)
            break
    
    if _validate(U.reshape(n*m, rank).tolist(), V.reshape(m*p, rank).tolist(), W.reshape(p*n, rank).tolist(), n, m, p):
        return U.reshape(n*m, rank), V.reshape(m*p, rank), W.reshape(p*n, rank)
    
    return None
