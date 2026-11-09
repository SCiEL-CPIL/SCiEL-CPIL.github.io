def construct():
    n, m, p = 2, 4, 6
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 12 terms for k in 0, 1, 2
    for j in range(m):
        for k in range(3):
            r = j * 3 + k
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
                
    # Next 12 terms for k in 3, 4, 5
    for j in range(m):
        for k in range(3, 6):
            r = 12 + j * 3 + (k - 3)
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
