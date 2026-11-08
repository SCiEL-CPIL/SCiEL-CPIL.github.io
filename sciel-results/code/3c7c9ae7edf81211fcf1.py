def construct():
    n, m, p = 2, 3, 6
    r = m * p  # 18
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[j * p + k][c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
