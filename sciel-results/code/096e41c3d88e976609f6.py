def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, col j. Shape (n*m, rank).
                # A is n x m. Index in U: i*m + j
                U[i*m + j][col] = 1
                # V index: row j of B, col k. Shape (m*p, rank).
                # B is m x p. Index in V: j*p + k
                V[j*p + k][col] = 1
                # W index: row k of C, col i. Shape (p*n, rank).
                # C is p x n. Trace convention W[k*n + i, r]
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
