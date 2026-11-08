<<<<<<< SEARCH
    n, m, p = DIMENSIONS
    rank = 6
    U = [[0] * rank for _ in range(n * m)]  # 6 x 6
    V = [[0] * rank for _ in range(m * p)]  # 27 x 6
    W = [[0] * rank for _ in range(p * n)]  # 18 x 6
    
    for j in range(m):
        for i in range(n):
            r = j * n + i
            # U: index i*m + j
            U[i * m + j][r] = 1
            # V: indices j*p + k for k=0..p-1
            for k in range(p):
                V[j * p + k][r] = 1
            # W: indices k*n + i for k=0..p-1
            for k in range(p):
                W[k * n + i][r] = 1
=======
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]  # 6 x 3
    V = [[0] * rank for _ in range(m * p)]  # 27 x 3
    W = [[0] * rank for _ in range(p * n)]  # 18 x 3
    
    for j in range(m):
        # U: index i*m + j, column j
        for i in range(n):
            U[i * m + j][j] = 1
        # V: indices j*p + k for k=0..p-1, column j
        for k in range(p):
            V[j * p + k][j] = 1
        # W: indices k*n + i for k=0..p-1, column j
        # W is 1 for all i, k for this j
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
>>>>>>> REPLACE
