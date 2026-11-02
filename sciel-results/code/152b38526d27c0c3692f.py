def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 8 = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U: for both i values
            U[0 * m + j][c] = 1
            U[1 * m + j][c] = 1
            # V: for this j,k
            V[j * p + k][c] = 1
            # W: for both i values
            W[k * n + 0][c] = 1
            W[k * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}
