import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55: 5 * 7 (Strassen 2x2x2) + 20 (2x1x10)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2x2x2
    for b in range(5):
        # k indices for this block
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for 2x2 matrix multiplication
        # A is 2x2 (rows 0,1; cols 0,1 of global A)
        # B is 2x2 (rows 0,1 of global B; cols k0, k1)
        # C is 2x2 (rows 0,1 of global C; cols k0, k1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    # 2x1x10 part
    # A col 2 (index 2), B row 2 (index 2)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
