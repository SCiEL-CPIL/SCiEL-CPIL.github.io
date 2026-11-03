def construct():
    n, m, p = 4, 4, 9
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for a in range(n):  # a = 0..3, corresponds to row of A / i index
        for b in range(m):  # b = 0..3, corresponds to col of A / j index
            c = a * m + b
            # U[i*m+j, c] = 1 if i==a and j==b
            U[a * m + b][c] = 1
            # V[j*p+k, c] = 1 if j==b (for all k)
            for k in range(p):
                V[b * p + k][c] = 1
            # W[k*n+i, c] = 1 if i==a (for all k)
            for k in range(p):
                W[k * n + a][c] = 1
    
    return {"U": U, "V": V, "W": W}
