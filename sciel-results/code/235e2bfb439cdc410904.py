def construct():
    n, m, p = 4, 5, 5
    rank = m  # 5
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for j in range(m):
        r = j
        # U: for all i, U[i*m + j, r] = 1
        for i in range(n):
            U[i*m + j, r] = 1
        # V: for all k, V[j*p + k, r] = 1
        for k in range(p):
            V[j*p + k, r] = 1
        # W: for all i,k, W[k*n + i, r] = 1
        for k in range(p):
            for i in range(n):
                W[k*n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
