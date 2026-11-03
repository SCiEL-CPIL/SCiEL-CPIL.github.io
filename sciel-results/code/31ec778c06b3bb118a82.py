DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):  # j0 = 0, 1, 2
        # U: 1 at positions (i*m + j0) for all i
        for i in range(n):
            U[i * m + j0][j0] = 1
        # V: 1 at positions (j0*p + k) for all k
        for k in range(p):
            V[j0 * p + k][j0] = 1
        # W: 1 at all positions (k*n + i)
        for k in range(p):
            for i in range(n):
                W[k * n + i][j0] = 1
    
    return {"U": U, "V": V, "W": W}
