def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25 for 2x5x5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: non-zero at i*m+j for i=0..n-1
            for i in range(n):
                U[i * m + j][col] = 1
            # V: non-zero at j*p+k
            V[j * p + k][col] = 1
            # W: non-zero at k*n+i for i=0..n-1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
