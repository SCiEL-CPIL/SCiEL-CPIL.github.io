def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 Strassen blocks
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11
        U[0][col] = 1
        U[3][col] = 1
        # V: B00, B11
        V[0 * p + k_start][col] = 1
        V[1 * p + (k_start + 1)][col] = 1
        # W: C00, C11
        W[k_start * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[2][col] = 1
        U[3][col] = 1
        V[0 * p + k_start][col] = 1
        W[k_start * n + 1][col] = 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[0][col] = 1
        V[0 * p + (k_start + 1)][col] = 1
        V[1 * p + (k_start + 1)][col] = -1
        W[(k_start + 1) * n + 0][col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[3][col] = 1
        V[1 * p + k_start][col] = 1
        V[0 * p + k_start][col] = -1
        W[k_start * n + 0][col] = 1
        W[k_start * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + (k_start + 1)][col] = 1
        W[k_start * n + 0][col] = -1
        W[(k_start + 1) * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0 * p + k_start][col] = 1
        V[0 * p + (k_start + 1)][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1 * p + k_start][col] = 1
        V[1 * p + (k_start + 1)][col] = 1
        W[k_start * n + 0][col] = 1
        col += 1

    # Last column k=8
    k = 8
    
    # M1 = A00 B0k
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # M2 = A01 B1k
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # M3 = A10 B0k
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    # M4 = A11 B1k
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
