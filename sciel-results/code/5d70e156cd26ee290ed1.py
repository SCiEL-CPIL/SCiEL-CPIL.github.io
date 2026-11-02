<<<<<<< SEARCH
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
=======
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Quick verification
    if time.time() < _deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(0)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_check = np.zeros((n, p))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            C_check += (u @ v) * w.T  # w.T[i,j] = w[j,i] = W[j*n+i, r]
        
        if np.max(np.abs(C_check - C_expected)) > 1e-10:
            raise RuntimeError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
