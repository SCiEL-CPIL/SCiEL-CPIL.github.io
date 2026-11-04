import time

def construct():
    n, m, p = 3, 3, 8
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i of A (0, 1, 2), we use 3 ranks
    for i in range(n):
        base_r = i * m
        for j in range(m):
            r = base_r + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
