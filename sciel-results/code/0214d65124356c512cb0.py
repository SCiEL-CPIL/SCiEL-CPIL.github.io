DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for k in range(p):
            r = i * p + k
            W[k*n + i][r] = 1
            for j in range(m):
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
    return {"U": U, "V": V, "W": W}
