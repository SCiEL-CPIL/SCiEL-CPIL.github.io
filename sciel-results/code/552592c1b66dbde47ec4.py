import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition
    # Split inner dimension m=5 into 2, 2, 1
    # A is 2x5, B is 5x5
    # A1 (2x2), A2 (2x2), A3 (2x1)
    # B1 (2x5), B2 (2x5), B3 (1x5)
    
    # We need to build U, V, W
    # U shape (n*m, rank) = (10, rank)
    # V shape (m*p, rank) = (25, rank)
    # W shape (p*n, rank) = (10, rank)
    
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Block 1: A1 (cols 0,1), B1 (rows 0,1)
    # A1 indices in U: i in 0..1, j in 0..1 -> U[i*5 + j]
    # B1 indices in V: j in 0..1, k in 0..4 -> V[j*5 + k]
    # Output C1: i in 0..1, k in 0..4 -> W[k*2 + i]
    
    # Block 2: A2 (cols 2,3), B2 (rows 2,3)
    # A2 indices in U: i in 0..1, j in 2..3 -> U[i*5 + j]
    # B2 indices in V: j in 2..3, k in 0..4 -> V[j*5 + k]
    
    # Block 3: A3 (col 4), B3 (row 4)
    # A3 indices in U: i in 0..1, j in 4 -> U[i*5 + 4]
    # B3 indices in V: j in 4, k in 0..4 -> V[4*5 + k]
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_2x2x2(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, w_row_offsets, w_col_offsets):
        nonlocal col_offset
        # a_row_offsets: list of 2 row indices for A
        # a_col_offsets: list of 2 col indices for A
        # b_row_offsets: list of 2 row indices for B
        # b_col_offsets: list of 2 col indices for B
        # w_row_offsets: list of 2 row indices for C (k)
        # w_col_offsets: list of 2 col indices for C (i)
        
        # Strassen terms
        # M1 = (a00)(b00 + b01)
        # M2 = (a10 + a11)(b00)
        # M3 = (a00 + a01)(b11 - b10)
        # M4 = (a11)(b10 - b11)
        # M5 = (a00 + a11)(b00 + b11)
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # Note: Standard Strassen:
        # M1 = a11 * (b11 + b12)
        # M2 = (a21 + a22) * b11
        # M3 = (a11 + a12) * (b22 - b21)
        # M4 = a22 * (b21 - b22)
        # M5 = (a11 + a22) * (b11 + b22)
        # M6 = (a21 - a11) * (b11 + b12)
        # M7 = (a12 - a22) * (b21 + b22)
        
        # Let's map indices carefully.
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # M1: a00 * (b00 + b01)
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][c] = 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][c] = 1
        V[b_row_offsets[0] * p + b_col_offsets[1]][c] = 1
        # C11 += M1, C12 += M1
        W[w_row_offsets[0] * n + w_col_offsets[0]][c] = 1
        W[w_row_offsets[0] * n + w_col_offsets[1]][c] = 1
        
        # M2: (a10 + a11) * b00
        c = col_offset; col_offset += 1
        U[a_row_offsets[1] * m + a_col_offsets[0]][c] = 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][c] = 1
        # C21 += M2, C22 += M2 (Wait, C22 is M5+M1-M2-M4. C21 is M2+M3)
        # Let's use standard formulas:
        # C11 = M5 + M4 - M6 - M7
        # C12 = M1 + M7
        # C21 = M2 + M3
        # C22 = M5 + M1 - M2 - M4
        
        W[w_row_offsets[1] * n + w_col_offsets[0]][c] = 1
        W[w_row_offsets[1] * n + w_col_offsets[1]][c] = -1
        
        # M3: (a00 + a01) * (b11 - b10)
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][c] = 1
        U[a_row_offsets[0] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[0]][c] = -1
        W[w_row_offsets[1] * n + w_col_offsets[0]][c] = 1
        W[w_row_offsets[1] * n + w_col_offsets[1]][c] = 1
        
        # M4: a11 * (b10 - b11)
        c = col_offset; col_offset += 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[0]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][c] = -1
        W[w_row_offsets[0] * n + w_col_offsets[0]][c] = 1
        W[w_row_offsets[1] * n + w_col_offsets[1]][c] = -1
        
        # M5: (a00 + a11) * (b00 + b11)
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][c] = 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][c] = 1
        W[w_row_offsets[0] * n + w_col_offsets[0]][c] = 1
        W[w_row_offsets[1] * n + w_col_offsets[1]][c] = 1
        
        # M6: (a10 - a00) * (b00 + b01)
        c = col_offset; col_offset += 1
        U[a_row_offsets[1] * m + a_col_offsets[0]][c] = 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][c] = -1
        V[b_row_offsets[0] * p + b_col_offsets[0]][c] = 1
        V[b_row_offsets[0] * p + b_col_offsets[1]][c] = 1
        W[w_row_offsets[0] * n + w_col_offsets[0]][c] = -1
        
        # M7: (a01 - a11) * (b10 + b11)
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[1]][c] = 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][c] = -1
        V[b_row_offsets[1] * p + b_col_offsets[0]][c] = 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][c] = 1
        W[w_row_offsets[0] * n + w_col_offsets[1]][c] = 1
        W[w_row_offsets[1] * n + w_col_offsets[0]][c] = -1
        
    # Helper to add schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offset, w_row_offset, w_col_offsets):
        nonlocal col_offset
        # A is 2x2, B is 2x1, C is 2x1
        # C1 = a00*b0 + a01*b1
        # C2 = a10*b0 + a11*b1
        
        # Term 1: a00 * b0
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][c] = 1
        V[b_row_offsets[0] * p + b_col_offset][c] = 1
        W[w_row_offset * n + w_col_offsets[0]][c] = 1
        
        # Term 2: a01 * b1
        c = col_offset; col_offset += 1
        U[a_row_offsets[0] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[1] * p + b_col_offset][c] = 1
        W[w_row_offset * n + w_col_offsets[0]][c] = 1
        
        # Term 3: a10 * b0
        c = col_offset; col_offset += 1
        U[a_row_offsets[1] * m + a_col_offsets[0]][c] = 1
        V[b_row_offsets[0] * p + b_col_offset][c] = 1
        W[w_row_offset * n + w_col_offsets[1]][c] = 1
        
        # Term 4: a11 * b1
        c = col_offset; col_offset += 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][c] = 1
        V[b_row_offsets[1] * p + b_col_offset][c] = 1
        W[w_row_offset * n + w_col_offsets[1]][c] = 1
        
    # Block 1: A cols 0,1; B rows 0,1
    # A1 rows 0,1
    a_rows = [0, 1]
    a_cols = [0, 1]
    b_rows = [0, 1]
    
    # B cols 0,1 -> W rows 0,1
    add_strassen_2x2x2(a_rows, a_cols, b_rows, [0, 1], [0, 1], [0, 1])
    # B cols 2,3 -> W rows 2,3
    add_strassen_2x2x2(a_rows, a_cols, b_rows, [2, 3], [2, 3], [0, 1])
    # B col 4 -> W row 4
    add_schoolbook_2x2x1(a_rows, a_cols, b_rows, 4, 4, [0, 1])
    
    # Block 2: A cols 2,3; B rows 2,3
    a_cols = [2, 3]
    b_rows = [2, 3]
    
    add_strassen_2x2x2(a_rows, a_cols, b_rows, [0, 1], [0, 1], [0, 1])
    add_strassen_2x2x2(a_rows, a_cols, b_rows, [2, 3], [2, 3], [0, 1])
    add_schoolbook_2x2x1(a_rows, a_cols, b_rows, 4, 4, [0, 1])
    
    # Block 3: A col 4; B row 4
    a_cols = [4]
    b_rows = [4]
    # A is 2x1, B is 1x5
    # C is 2x5
    # Schoolbook 2x1x5
    for k in range(5):
        c = col_offset; col_offset += 1
        # i=0
        U[0 * m + 4][c] = 1
        V[4 * p + k][c] = 1
        W[k * n + 0][c] = 1
        # i=1
        c = col_offset; col_offset += 1
        U[1 * m + 4][c] = 1
        V[4 * p + k][c] = 1
        W[k * n + 1][c] = 1

    return {"U": U, "V": V, "W": W}
