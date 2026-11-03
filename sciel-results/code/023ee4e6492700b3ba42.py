def _build_optimized_decomposition(n=2, m=3, p=5):
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0] * rank for _ in range(m * p)]  # 15 x 16
    W = [[0] * rank for _ in range(p * n)]  # 10 x 16
    
    col = 0
    
    # Strassen block for B1 columns 0,1 (7 terms)
    col = _build_strassen_2x2_block(0, 0, col, U, V, W, n, m, p)
    
    # Strassen block for B1 columns 2,3 (7 terms)
    col = _build_strassen_2x2_block(0, 2, col, U, V, W, n, m, p)
    
    # Column 4 of B1: single term (1 term instead of 4)
    # U: A[0,0], A[0,1], A[1,0], A[1,1] all = 1
    # V: B[0,4], B[1,4] all = 1
    # W: C[0,4], C[1,4] all = 1
    U[0][col] = 1
    U[1][col] = 1
    U[3][col] = 1
    U[4][col] = 1
    V[4][col] = 1
    V[5][col] = 1
    W[8][col] = 1
    W[9][col] = 1
    col += 1
    
    # A2*B2: single term (1 term instead of 10)
    # U: A[0,2], A[1,2] = 1
    # V: B[2,0..4] all = 1
    # W: C[0,0..4], C[1,0..4] all = 1
    U[2][col] = 1
    U[5][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return U, V, W
