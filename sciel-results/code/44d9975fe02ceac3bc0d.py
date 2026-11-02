def construct():
    n, m, p = 2, 2, 6
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for l in range(m):
            r = k * m + l
            # U: index i*m + l
            for i in range(n):
                U[i * m + l][r] = 1
            # V: index l*p + k
            V[l * p + k][r] = 1
            # W: index k*n + i
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
