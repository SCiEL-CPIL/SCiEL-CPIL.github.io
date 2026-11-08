"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols):
        nonlocal col
        # a_rows: 2 indices for A rows (0, 1)
        # a_cols: 2 indices for A cols
        # b_rows: 2 indices for B rows
        # b_cols: 2 indices for B cols
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Local indices 0,1 for A and B sub-matrices
        # Global A index: a_rows[i] * m + a_cols[j]
        # Global B index: b_rows[i] * p + b_cols[j]
        # Global W index: b_cols[j] * n + a_rows[i]
        
        # M1
        # A part: A00, A11
        U[a_rows[0] * m + a_cols[0]][col] = 1
        U[a_rows[1] * m + a_cols[1]][col] = 1
        # B part: B00, B11
        V[b_rows[0] * p + b_cols[0]][col] = 1
        V[b_rows[1] * p + b_cols[1]][col] = 1
        # W part: C00 += M1, C11 += M1
        W[b_cols[0] * n + a_rows[0]][col] = 1
        W[b_cols[1] * n + a_rows[1]][col] = 1
        col += 1
        
        # M2
        # A part: A10, A11
        U[a_rows[1] * m + a_cols[0]][col] = 1
        U[a_rows[1] * m + a_cols[1]][col] = 1
        # B part: B00
        V[b_rows[0] * p + b_cols[0]][col] = 1
        # W part: C10 += M2, C11 -= M2
        W[b_cols[0] * n + a_rows[1]][col] = 1
        W[b_cols[1] * n + a_rows[1]][col] = -1
        col += 1
        
        # M3
        # A part: A00
        U[a_rows[0] * m + a_cols[0]][col] = 1
        # B part: B01, -B11
        V[b_rows[0] * p + b_cols[1]][col] = 1
        V[b_rows[1] * p + b_cols[1]][col] = -1
        # W part: C00 += M3, C01 += M3
        W[b_cols[0] * n + a_rows[0]][col] = 1
        W[b_cols[1] * n + a_rows[0]][col] = 1
        col += 1
        
        # M4
        # A part: A11
        U[a_rows[1] * m + a_cols[1]][col] = 1
        # B part: B10, -B00
        V[b_rows[1] * p + b_cols[0]][col] = 1
        V[b_rows[0] * p + b_cols[0]][col] = -1
        # W part: C00 += M4, C10 += M4
        W[b_cols[0] * n + a_rows[0]][col] = 1
        W[b_cols[0] * n + a_rows[1]][col] = 1
        col += 1
        
        # M5
        # A part: A00, A01
        U[a_rows[0] * m + a_cols[0]][col] = 1
        U[a_rows[0] * m + a_cols[1]][col] = 1
        # B part: B11
        V[b_rows[1] * p + b_cols[1]][col] = 1
        # W part: C00 -= M5, C01 += M5
        W[b_cols[0] * n + a_rows[0]][col] = -1
        W[b_cols[1] * n + a_rows[0]][col] = 1
        col += 1
        
        # M6
        # A part: A10, -A00
        U[a_rows[1] * m + a_cols[0]][col] = 1
        U[a_rows[0] * m + a_cols[0]][col] = -1
        # B part: B00, B01
        V[b_rows[0] * p + b_cols[0]][col] = 1
        V[b_rows[0] * p + b_cols[1]][col] = 1
        # W part: C11 += M6
        W[b_cols[1] * n + a_rows[1]][col] = 1
        col += 1
        
        # M7
        # A part: A01, -A11
        U[a_rows[0] * m + a_cols[1]][col] = 1
        U[a_rows[1] * m + a_cols[1]][col] = -1
        # B part: B10, B11
        V[b_rows[1] * p + b_cols[0]][col] = 1
        V[b_rows[1] * p + b_cols[1]][col] = 1
        # W part: C00 += M7
        W[b_cols[0] * n + a_rows[0]][col] = 1
        col += 1

    def add_standard_block(a_rows, a_cols, b_rows, b_col):
        nonlocal col
        # a_rows: 2 indices
        # a_cols: 2 indices
        # b_rows: 2 indices
        # b_col: 1 index
        
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # M1: A00 * B00 -> C00
        U[a_rows[0] * m + a_cols[0]][col] = 1
        V[b_rows[0] * p + b_col][col] = 1
        W[b_col * n + a_rows[0]][col] = 1
        col += 1
        
        # M2: A01 * B10 -> C00
        U[a_rows[0] * m + a_cols[1]][col] = 1
        V[b_rows[1] * p + b_col][col] = 1
        W[b_col * n + a_rows[0]][col] = 1
        col += 1
        
        # M3: A10 * B00 -> C10
        U[a_rows[1] * m + a_cols[0]][col] = 1
        V[b_rows[0] * p + b_col][col] = 1
        W[b_col * n + a_rows[1]][col] = 1
        col += 1
        
        # M4: A11 * B10 -> C10
        U[a_rows[1] * m + a_cols[1]][col] = 1
        V[b_rows[1] * p + b_col][col] = 1
        W[b_col * n + a_rows[1]][col] = 1
        col += 1

    # Block 1: A1 B11
    # A1: rows 0,1; cols 0,1
    # B11: rows 0,1; cols 0,1
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A1 B12
    # A1: rows 0,1; cols 0,1
    # B12: rows 0,1; cols 2,3
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3])
    
    # Block 3: A2 B21
    # A2: rows 0,1; cols 2,3
    # B21: rows 2,3; cols 0,1
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1])
    
    # Block 4: A2 B22
    # A2: rows 0,1; cols 2,3
    # B22: rows 2,3; cols 2,3
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3])
    
    # Block 5: A1 B13
    # A1: rows 0,1; cols 0,1
    # B13: rows 0,1; col 4
    add_standard_block([0, 1], [0, 1], [0, 1], 4)
    
    # Block 6: A2 B23
    # A2: rows 0,1; cols 2,3
    # B23: rows 2,3; col 4
    add_standard_block([0, 1], [2, 3], [2, 3], 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
