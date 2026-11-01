def construct():
    n, m, p = 3, 3, 7
    rank = n * p # 21
    # U: (n*m) x rank -> 9 x 21
    # V: (m*p) x rank -> 21 x 21
    # W: (p*n) x rank -> 21 x 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each k in 0..p-1
    for k in range(p):
        # For each i in 0..n-1
        for i in range(n):
            c = k * n + i # Column index: 0..20
            # U row: i*m + j. For all j, U[i*m+j, c] = 1
            for j in range(m):
                U[i * m + j][c] = 1
            # V row: j*p + k. For all j, V[j*p+k, c] = 1
            for j in range(m):
                V[j * p + k][c] = 1
            # W row: k*n + i. W[k*n+i, c] = 1
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
