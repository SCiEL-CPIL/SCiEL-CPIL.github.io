DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
