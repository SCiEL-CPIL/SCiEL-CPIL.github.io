DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            # U[i*m+j, c] = delta_{j,c}
            U[i * m + j][j] = 1  # c=j
    
    for j in range(m):
        for k in range(p):
            # V[j*p+k, c] = delta_{j,c}
            V[j * p + k][j] = 1  # c=j
    
    for i in range(n):
        for k in range(p):
            # W[k*n+i, c] = 1 for all c
            for c in range(rank):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
