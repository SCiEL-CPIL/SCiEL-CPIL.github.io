def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = (A @ B).reshape(-1)  # (p*n,) with C_flat[k*n+i] = C[i,k]
        
        A_flat = A.reshape(-1)  # (n*m,)
        B_flat = B.reshape(-1)  # (m*p,)
        
        C_computed = np.zeros(p * n)
        for r in range(U.shape[1]):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            C_computed += u_val * v_val * W[:, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True
