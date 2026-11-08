def hooi_step(U, V, W, T_flat, n, m, p):
    """
    Perform one HOOI/ALS step.
    U: (nm, R)
    V: (mp, R)
    W: (pn, R)
    T_flat: (nmp,) tensor values
    """
    R = U.shape[1]
    
    # Update U
    # A_U shape: (nmp, nm * R)
    # Index mapping:
    # Row: (i, j, k) -> i * m * p + j * p + k
    # Col: (ij, t) -> (i * m + j) * R + t
    A_U = np.zeros((n * m * p, n * m * R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for t in range(R):
                    col = (i * m + j) * R + t
                    # Coefficient for U[i*m+j, t] is V[j*p+k, t] * W[k*n+i, t]
                    A_U[row, col] = V[j * p + k, t] * W[k * n + i, t]
    
    # Solve A_U * u = T_flat
    u, residuals, rank, s = np.linalg.lstsq(A_U, T_flat, rcond=None)
    U = u.reshape((n * m, R))
    
    # Update V
    A_V = np.zeros((n * m * p, m * p * R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for t in range(R):
                    col = (j * p + k) * R + t
                    # Coefficient for V[j*p+k, t] is U[i*m+j, t] * W[k*n+i, t]
                    A_V[row, col] = U[i * m + j, t] * W[k * n + i, t]
                    
    v, residuals, rank, s = np.linalg.lstsq(A_V, T_flat, rcond=None)
    V = v.reshape((m * p, R))
    
    # Update W
    A_W = np.zeros((n * m * p, p * n * R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for t in range(R):
                    col = (k * n + i) * R + t
                    # Coefficient for W[k*n+i, t] is U[i*m+j, t] * V[j*p+k, t]
                    A_W[row, col] = U[i * m + j, t] * V[j * p + k, t]
                    
    w, residuals, rank, s = np.linalg.lstsq(A_W, T_flat, rcond=None)
    W = w.reshape((p * n, R))
    
    return U, V, W
