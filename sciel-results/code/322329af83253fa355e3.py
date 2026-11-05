def try_als_vectorized(rank, max_iter=1000, tol=1e-10, seed=42):
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        if time.time() > deadline:
            return None
        
        # Update W: for each (k,i), solve min ||U[i*3+j] * V[j*4+k] * W[k*2+i] - 1||
        # This is: for fixed (k,i), we have a linear system in W[k*2+i]
        # Σ_j U[i*3+j, t] * V[j*4+k, t] * W[k*2+i, t] = 1 for all t
        # This is: A_W @ W_row = ones, where A_W[j,t] = U[i*3+j,t] * V[j*4+k,t]
        
        W_new = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                A_w = np.zeros((m, rank))
                for j in range(m):
                    A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                W_new[k*n+i, :] = np.linalg.lstsq(A_w, np.ones(m), rcond=None)[0]
        
        # Similarly for U and V
        U_new = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                A_u = np.zeros((p, rank))
                for k in range(p):
                    A_u[k, :] = V[j*p+k, :] * W[k*n+i, :]
                U_new[i*m+j, :] = np.linalg.lstsq(A_u, np.ones(p), rcond=None)[0]
        
        V_new = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                A_v = np.zeros((n, rank))
                for i in range(n):
                    A_v[i, :] = U[i*m+j, :] * W[k*n+i, :]
                V_new[j*p+k, :] = np.linalg.lstsq(A_v, np.ones(n), rcond=None)[0]
        
        # Check convergence
        diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
        W, U, V = W_new, U_new, V_new
        if diff < tol:
            break
    
    # Verify
    ...
