def construct():
    n, m, p = 2, 4, 5
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            s = i * p + k
            for j in range(m):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][s] = 1
                V[b][s] = 1
                W[c][s] = 1
    
    return {"U": U, "V": V, "W": W}
