DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    R = n * m  # 16
    U = [[0] * R for _ in range(n * m)]  # 16 x 16
    V = [[0] * R for _ in range(m * p)]  # 32 x 16
    W = [[0] * R for _ in range(p * n)]  # 32 x 16
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
