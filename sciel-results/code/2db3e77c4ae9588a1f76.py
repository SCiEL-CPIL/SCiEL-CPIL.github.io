def construct():
    n, m, p = 2, 3, 6
    rank = m * p  # 18
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        j = c // p
        k = c % p
        # U[i*m + j, c] = 1 for all i
        for i in range(n):
            U[i * m + j, c] = 1.0
        # V[j*p + k, c] = 1
        V[j * p + k, c] = 1.0
        # W[k*n + i, c] = 1 for all i
        for i in range(n):
            W[k * n + i, c] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
