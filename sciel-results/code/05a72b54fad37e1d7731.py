import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    
    # Initialize U, V, W as lists of lists for easy manipulation, or numpy arrays
    # U: (n*m, rank) -> (8, 50)
    # V: (m*p, rank) -> (28, 50)
    # W: (p*n, rank) -> (14, 50)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, c_row_offsets, c_col_offsets):
        nonlocal col
        # Local indices:
        # A: 2x2, rows a_row_offsets[0..1], cols a_col_offsets[0..1]
        # B: 2x2, rows b_row_offsets[0..1], cols b_col_offsets[0..1]
        # C: 2x2, rows c_row_offsets[0..1], cols c_col_offsets[0..1]
        
        # 7 multiplications
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Reconstruction:
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We define the coefficients for A, B, C for each multiplication
        
        # U coefficients (A)
        # V coefficients (B)
        # W coefficients (C)
        
        # m1
        U[a_row_offsets[0]*m + a_col_offsets[0]][col] += 1
        U[a_row_offsets[1]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[0]*p + b_col_offsets[0]][col] += 1
        V[b_row_offsets[1]*p + b_col_offsets[1]][col] += 1
        W[c_col_offsets[0]*n + c_row_offsets[0]][col] += 1 # c00
        W[c_col_offsets[1]*n + c_row_offsets[1]][col] += 1 # c11
        col += 1
        
        # m2
        U[a_row_offsets[1]*m + a_col_offsets[0]][col] += 1
        U[a_row_offsets[1]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[0]*p + b_col_offsets[0]][col] += 1
        W[c_col_offsets[0]*n + c_row_offsets[1]][col] += 1 # c10
        col += 1
        
        # m3
        U[a_row_offsets[0]*m + a_col_offsets[0]][col] += 1
        V[b_row_offsets[0]*p + b_col_offsets[1]][col] += 1
        V[b_row_offsets[1]*p + b_col_offsets[1]][col] -= 1
        W[c_col_offsets[1]*n + c_row_offsets[0]][col] += 1 # c01
        col += 1
        
        # m4
        U[a_row_offsets[1]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[1]*p + b_col_offsets[0]][col] += 1
        V[b_row_offsets[0]*p + b_col_offsets[0]][col] -= 1
        W[c_col_offsets[0]*n + c_row_offsets[0]][col] += 1 # c00
        W[c_col_offsets[0]*n + c_row_offsets[1]][col] += 1 # c10
        col += 1
        
        # m5
        U[a_row_offsets[0]*m + a_col_offsets[0]][col] += 1
        U[a_row_offsets[0]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[1]*p + b_col_offsets[1]][col] += 1
        W[c_col_offsets[0]*n + c_row_offsets[0]][col] -= 1 # c00
        W[c_col_offsets[1]*n + c_row_offsets[0]][col] += 1 # c01
        col += 1
        
        # m6
        U[a_row_offsets[1]*m + a_col_offsets[0]][col] += 1
        U[a_row_offsets[0]*m + a_col_offsets[0]][col] -= 1
        V[b_row_offsets[0]*p + b_col_offsets[0]][col] += 1
        V[b_row_offsets[0]*p + b_col_offsets[1]][col] += 1
        W[c_col_offsets[1]*n + c_row_offsets[1]][col] += 1 # c11
        col += 1
        
        # m7
        U[a_row_offsets[0]*m + a_col_offsets[1]][col] += 1
        U[a_row_offsets[1]*m + a_col_offsets[1]][col] -= 1
        V[b_row_offsets[1]*p + b_col_offsets[0]][col] += 1
        V[b_row_offsets[1]*p + b_col_offsets[1]][col] += 1
        W[c_col_offsets[0]*n + c_row_offsets[0]][col] += 1 # c00
        col += 1

    # Helper to add a Schoolbook block (2x2x1)
    def add_schoolbook_block_2x2x1(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offset, c_row_offsets, c_col_offset):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # c0 = a00*b00 + a01*b10
        # c1 = a10*b00 + a11*b10
        
        # m1 = a00*b00
        U[a_row_offsets[0]*m + a_col_offsets[0]][col] += 1
        V[b_row_offsets[0]*p + b_col_offset][col] += 1
        W[c_col_offset*n + c_row_offsets[0]][col] += 1
        col += 1
        
        # m2 = a01*b10
        U[a_row_offsets[0]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[1]*p + b_col_offset][col] += 1
        W[c_col_offset*n + c_row_offsets[0]][col] += 1
        col += 1
        
        # m3 = a10*b00
        U[a_row_offsets[1]*m + a_col_offsets[0]][col] += 1
        V[b_row_offsets[0]*p + b_col_offset][col] += 1
        W[c_col_offset*n + c_row_offsets[1]][col] += 1
        col += 1
        
        # m4 = a11*b10
        U[a_row_offsets[1]*m + a_col_offsets[1]][col] += 1
        V[b_row_offsets[1]*p + b_col_offset][col] += 1
        W[c_col_offset*n + c_row_offsets[1]][col] += 1
        col += 1

    # Process first half of m (j=0,1)
    # A1: rows 0,1; cols 0,1
    # B1: rows 0,1; cols 0..6
    # C: rows 0,1; cols 0..6
    
    a_rows = [0, 1]
    a_cols_1 = [0, 1]
    b_rows_1 = [0, 1]
    
    # B1 cols: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Schoolbook)
    add_strassen_block(a_rows, a_cols_1, b_rows_1, [0, 1], a_rows, [0, 1])
    add_strassen_block(a_rows, a_cols_1, b_rows_1, [2, 3], a_rows, [2, 3])
    add_strassen_block(a_rows, a_cols_1, b_rows_1, [4, 5], a_rows, [4, 5])
    add_schoolbook_block_2x2x1(a_rows, a_cols_1, b_rows_1, 6, a_rows, 6)
    
    # Process second half of m (j=2,3)
    # A2: rows 0,1; cols 2,3
    # B2: rows 2,3; cols 0..6
    # C: rows 0,1; cols 0..6
    
    a_cols_2 = [2, 3]
    b_rows_2 = [2, 3]
    
    add_strassen_block(a_rows, a_cols_2, b_rows_2, [0, 1], a_rows, [0, 1])
    add_strassen_block(a_rows, a_cols_2, b_rows_2, [2, 3], a_rows, [2, 3])
    add_strassen_block(a_rows, a_cols_2, b_rows_2, [4, 5], a_rows, [4, 5])
    add_schoolbook_block_2x2x1(a_rows, a_cols_2, b_rows_2, 6, a_rows, 6)
    
    return {"U": U, "V": V, "W": W}
