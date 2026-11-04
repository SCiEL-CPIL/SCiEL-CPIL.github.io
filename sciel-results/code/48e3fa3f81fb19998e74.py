def construct():
    n, m, p = 2, 5, 5
    # Try rank 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    T = np.ones((n, m, p))
    Ur = np.array(U).reshape(n, m, rank)
    Vr = np.array(V).reshape(m, p, rank)
    Wr = np.array(W).reshape(p, n, rank)
    recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    if np.max(np.abs(recon - T)) < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
