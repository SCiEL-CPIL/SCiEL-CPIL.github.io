import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m rows, V: m*p rows, W: p*n rows
    # Rank 19
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 B1 (Rank 1)
    # A1 is col 0 of A (2x1). B1 is row 0 of B (1x5).
    # C_ij = A_i0 B_0j
    # U[i*3+0, col] = 1
    # V[0*5+j, col] = 1
    # W[j*2+i, col] = 1
    for i in range(n):
        U[i * m + 0][col] = 1
    for j in range(p):
        V[0 * p + j][col] = 1
    for j in range(p):
        for i in range(n):
            W[j * n + i][col] = 1
    col += 1
    
    # Part 2: A2 B2a (Strassen, Rank 7)
    # A2 is cols 1,2 of A. B2a is rows 1,2 and cols 0,1 of B.
    # A indices: i in {0,1}, j in {1,2}. U index i*3+j.
    # B indices: k in {1,2}, l in {0,1}. V index k*5+l.
    # C indices: i in {0,1}, l in {0,1}. W index l*2+i.
    
    # Strassen terms for 2x2 by 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_strassen_block(a_col_offset, b_col_offset, b_row_offset, c_col_offset):
        nonlocal col
        # a_col_offset: 1
        # b_col_offset: 0 or 2
        # b_row_offset: 1
        # c_col_offset: 0 or 2
        
        # Local A indices: (0,0)->(0,1), (0,1)->(0,2), (1,0)->(1,1), (1,1)->(1,2)
        # Local B indices: (0,0)->(1,0), (0,1)->(1,1), (1,0)->(2,0), (1,1)->(2,1)
        
        # M1
        U[0*m + 1][col] = 1; U[1*m + 2][col] = 1
        V[1*p + 0][col] = 1; V[2*p + 1][col] = 1
        W[0*n + 0][col] = 1; W[0*n + 1][col] = 1; W[1*n + 0][col] = 1; W[1*n + 1][col] = 1
        col += 1
        
        # M2
        U[1*m + 1][col] = 1; U[1*m + 2][col] = 1
        V[1*p + 0][col] = 1
        W[1*n + 0][col] = 1
        col += 1
        
        # M3
        U[0*m + 1][col] = 1
        V[2*p + 0][col] = 1; V[2*p + 1][col] = -1
        W[1*n + 1][col] = 1
        col += 1
        
        # M4
        U[1*m + 2][col] = 1
        V[1*p + 1][col] = 1; V[1*p + 0][col] = -1
        W[0*n + 1][col] = 1
        col += 1
        
        # M5
        U[0*m + 1][col] = 1; U[0*m + 2][col] = 1
        V[2*p + 1][col] = 1
        W[0*n + 0][col] = 1; W[1*n + 1][col] = 1
        col += 1
        
        # M6
        U[1*m + 1][col] = 1; U[0*m + 1][col] = -1
        V[1*p + 0][col] = 1; V[1*p + 1][col] = 1
        W[1*n + 1][col] = 1
        col += 1
        
        # M7
        U[0*m + 2][col] = 1; U[1*m + 2][col] = -1
        V[2*p + 0][col] = 1; V[2*p + 1][col] = 1
        W[0*n + 0][col] = 1; W[1*n + 0][col] = 1
        col += 1

    # B2a: b_col_offset=0
    add_strassen_block(1, 0, 1, 0)
    
    # Part 3: A2 B2b (Strassen, Rank 7)
    # B2b: rows 1,2, cols 2,3.
    add_strassen_block(1, 2, 1, 2)
    
    # Part 4: A2 B2c (Schoolbook, Rank 4)
    # B2c: rows 1,2, col 4.
    # C_i4 = A_i1 B_14 + A_i2 B_24
    
    # C04 = A01 B14 + A02 B24
    U[0*m + 1][col] = 1; V[1*p + 4][col] = 1; W[4*n + 0][col] = 1
    col += 1
    U[0*m + 2][col] = 1; V[2*p + 4][col] = 1; W[4*n + 0][col] = 1
    col += 1
    
    # C14 = A11 B14 + A12 B24
    U[1*m + 1][col] = 1; V[1*p + 4][col] = 1; W[4*n + 1][col] = 1
    col += 1
    U[1*m + 2][col] = 1; V[2*p + 4][col] = 1; W[4*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
