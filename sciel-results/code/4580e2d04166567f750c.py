import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 * 2x2
    # A indices: (i, j), B indices: (j, k), C indices: (i, k)
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # A is 2x2 at [a_i:a_i+2, a_j:a_j+2]
        # B is 2x2 at [b_j:b_j+2, b_k:b_k+2]
        # C is 2x2 at [c_i:c_i+2, c_k:c_k+2] -> Note: C index in W is k*n + i
        
        # M1
        U[a_i * m + a_j][col] = 1
        U[a_i * m + a_j + 1][col] = 1
        U[a_i + 1 * m + a_j][col] = 1
        U[a_i + 1 * m + a_j + 1][col] = 1
        V[b_j * p + b_k][col] = 1
        V[b_j * p + b_k + 1][col] = 1
        V[b_j + 1 * p + b_k][col] = 1
        V[b_j + 1 * p + b_k + 1][col] = 1
        # C11, C22
        W[c_k * n + c_i][col] = 1
        W[(c_k + 1) * n + (c_i + 1)][col] = 1
        col += 1
        
        # M2
        U[a_i + 1 * m + a_j][col] = 1
        U[a_i + 1 * m + a_j + 1][col] = 1
        V[b_j * p + b_k][col] = 1
        # C21, C22
        W[c_k * n + (c_i + 1)][col] = 1
        W[(c_k + 1) * n + (c_i + 1)][col] = -1
        col += 1
        
        # M3
        U[a_i * m + a_j][col] = 1
        V[b_j * p + (b_k + 1)][col] = 1
        V[(b_j + 1) * p + b_k][col] = -1
        # C12, C22
        W[(c_k + 1) * n + c_i][col] = 1
        W[(c_k + 1) * n + (c_i + 1)][col] = 1
        col += 1
        
        # M4
        U[a_i + 1 * m + a_j + 1][col] = 1
        V[(b_j + 1) * p + b_k][col] = 1
        V[b_j * p + b_k][col] = -1
        # C11, C21
        W[c_k * n + c_i][col] = 1
        W[c_k * n + (c_i + 1)][col] = 1
        col += 1
        
        # M5
        U[a_i * m + a_j][col] = 1
        U[a_i * m + (a_j + 1)][col] = 1
        V[(b_j + 1) * p + (b_k + 1)][col] = 1
        # C11, C12
        W[c_k * n + c_i][col] = -1
        W[(c_k + 1) * n + c_i][col] = 1
        col += 1
        
        # M6
        U[a_i + 1 * m + a_j][col] = 1
        U[a_i * m + a_j][col] = -1
        V[b_j * p + b_k][col] = 1
        V[b_j * p + (b_k + 1)][col] = 1
        # C22
        W[(c_k + 1) * n + (c_i + 1)][col] = 1
        col += 1
        
        # M7
        U[a_i * m + (a_j + 1)][col] = 1
        U[a_i + 1 * m + a_j + 1][col] = -1
        V[(b_j + 1) * p + b_k][col] = 1
        V[(b_j + 1) * p + (b_k + 1)][col] = 1
        # C11
        W[c_k * n + c_i][col] = 1
        col += 1

    # Block 1: A1 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-5)
    # B_top is 2x6. Split into three 2x2 blocks.
    # B block 1: rows 0-1, cols 0-1. Output C rows 0-1, cols 0-1.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # B block 2: rows 0-1, cols 2-3. Output C rows 0-1, cols 2-3.
    add_strassen_block(0, 0, 0, 2, 2, 0)
    # B block 3: rows 0-1, cols 4-5. Output C rows 0-1, cols 4-5.
    add_strassen_block(0, 0, 0, 4, 4, 0)
    
    # Block 2: A2 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 0-5)
    # B_bot is 2x6. Split into three 2x2 blocks.
    # B block 1: rows 2-3, cols 0-1. Output C rows 0-1, cols 0-1.
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # B block 2: rows 2-3, cols 2-3. Output C rows 0-1, cols 2-3.
    add_strassen_block(0, 2, 2, 2, 2, 0)
    # B block 3: rows 2-3, cols 4-5. Output C rows 0-1, cols 4-5.
    add_strassen_block(0, 2, 2, 4, 4, 0)
    
    return {"U": U, "V": V, "W": W}
