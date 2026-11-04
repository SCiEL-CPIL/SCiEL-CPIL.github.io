def _rank32_als():
    n, m, p = 4, 4, 4
    r = 32
    np.random.seed(42)
    
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    for iteration in range(1000):
        # Update U
        for i in range(n):
            for j in range(m):
                # Build A: 4×32 matrix where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, r))
                for k in range(p):
                    for rr in range(r):
                        A[k, rr] = V[j*p+k, rr] * W[k*n+i, rr]
                b = np.ones(p)
                # Solve: min ||A*U[i*m+j, :]^T - b||^2
                # U[i*m+j, :]^T = A^T*(A*A^T)^{-1}*b
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for rr in range(r):
                        A[i, rr] = U[i*m+j, rr] * W[k*n+i, rr]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    for rr in range(r):
                        A[j, rr] = U[i*m+j, rr] * V[j*p+k, rr]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None
