def construct():
    n, m, p = DIMENSIONS
    # Try Rank 30
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Group 1: A1*B1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Group 2: a*c^T
    for i in range(3):
        for k in range(3):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # Group 3: A1*b
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    # Group 4: a*d
    for i in range(3):
        U[i * m + 3][col] = 1
        V[3 * p + 3][col] = 1
        W[3 * n + i][col] = 1
        col += 1
        
    # Verify
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    return {"U": U, "V": V, "W": W}
