"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition using Strassen for 2x2x2 blocks
    # A is 2x4, B is 4x5, C is 2x5
    # Split A into A0 (cols 0-1) and A1 (cols 2-3)
    # Split B into B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, col 4)
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, col 4)
    # C0 = A0*B00 + A1*B10 (2x2)
    # C1 = A0*B01 + A1*B11 (2x2)
    # C2 = A0*B02 + A1*B12 (2x1)
    
    # Strassen rank 7 for 2x2x2
    # Schoolbook rank 4 for 2x2x1
    
    rank = 4 * 7 + 2 * 4  # 28 + 8 = 36
    U = [[0] * rank for _ in range(n * m)]  # 8x36
    V = [[0] * rank for _ in range(m * p)]  # 20x36
    W = [[0] * rank for _ in range(p * n)]  # 10x36
    
    col_offset = 0
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, col_start):
        # a_row_offset: 0
        # a_col_offset: 0 or 2
        # b_row_offset: 0 or 2
        # b_col_offset: 0, 2, 4
        # c_row_offset: 0
        # c_col_offset: 0, 2, 4
        # 
        # Local indices:
        # A: i in {0,1}, j in {0,1} -> global j = j + a_col_offset
        # B: j in {0,1}, k in {0,1} -> global j = j + b_row_offset, k = k + b_col_offset
        # C: i in {0,1}, k in {0,1} -> global k = k + c_col_offset
        
        # Strassen coefficients
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We map these to U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        c = col_start
        # (a00+a11)
        U[a_row_offset*m + (0+a_col_offset)][c] += 1
        U[a_row_offset*m + (1+a_col_offset)][c] += 1
        # (b00+b11)
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += 1
        V[(1+b_row_offset)*p + (1+b_col_offset)][c] += 1
        # C00 += M1
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += 1
        # C11 += M1
        W[(1+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # M2
        c = col_start
        # (a10+a11)
        U[a_row_offset*m + (1+a_col_offset)][c] += 1
        U[a_row_offset*m + (1+a_col_offset)][c] += 1 # Wait, a10 is row 1, col 0. a11 is row 1, col 1.
        # Correction: A is 2x2.
        # a10: i=1, j=0. a11: i=1, j=1.
        U[a_row_offset*m + (1+a_col_offset)][c] = 1 # a10
        U[a_row_offset*m + (2+a_col_offset)][c] = 1 # a11? No, m=2 for local? No, m=4 global.
        # Local A is 2x2.
        # i=1, j=0 -> global i=1, j=a_col_offset
        # i=1, j=1 -> global i=1, j=a_col_offset+1
        U[a_row_offset*m + (1+a_col_offset)][c] = 1
        U[a_row_offset*m + (1+a_col_offset+1)][c] = 1
        # b00
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += 1
        # C10 += M2
        W[(0+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # M3
        c = col_start
        # a00
        U[a_row_offset*m + (0+a_col_offset)][c] += 1
        # (b01-b11)
        V[(0+b_row_offset)*p + (1+b_col_offset)][c] += 1
        V[(1+b_row_offset)*p + (1+b_col_offset)][c] += -1
        # C01 += M3
        W[(1+c_col_offset)*n + (0+a_row_offset)][c] += 1
        # C11 += M3
        W[(1+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # M4
        c = col_start
        # a11
        U[a_row_offset*m + (1+a_col_offset+1)][c] += 1
        # (b10-b00)
        V[(1+b_row_offset)*p + (0+b_col_offset)][c] += 1
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += -1
        # C00 += M4
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += 1
        # C10 += M4
        W[(0+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # M5
        c = col_start
        # (a00+a01)
        U[a_row_offset*m + (0+a_col_offset)][c] += 1
        U[a_row_offset*m + (1+a_col_offset+1)][c] += 1 # a01 is i=0, j=1
        # b11
        V[(1+b_row_offset)*p + (1+b_col_offset)][c] += 1
        # C00 -= M5
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += -1
        # C01 += M5
        W[(1+c_col_offset)*n + (0+a_row_offset)][c] += 1
        col_start += 1
        
        # M6
        c = col_start
        # (a10-a00)
        U[a_row_offset*m + (1+a_col_offset)][c] += 1
        U[a_row_offset*m + (0+a_col_offset)][c] += -1
        # (b00+b01)
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += 1
        V[(0+b_row_offset)*p + (1+b_col_offset)][c] += 1
        # C11 += M6
        W[(1+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # M7
        c = col_start
        # (a01-a11)
        U[a_row_offset*m + (1+a_col_offset+1)][c] += 1 # a01
        U[a_row_offset*m + (1+a_col_offset+1)][c] += -1 # a11? No.
        # a01: i=0, j=1. Index: 0*m + 1+a_col_offset
        # a11: i=1, j=1. Index: 1*m + 1+a_col_offset
        U[a_row_offset*m + (1+a_col_offset)][c] += 1 # a01
        U[a_row_offset*m + (1+a_col_offset+1)][c] += -1 # a11
        # (b10+b11)
        V[(1+b_row_offset)*p + (0+b_col_offset)][c] += 1
        V[(1+b_row_offset)*p + (1+b_col_offset)][c] += 1
        # C00 += M7
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += 1
        col_start += 1
        
        return col_start

    def add_schoolbook_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, col_start):
        # 2x2x1 block
        # A: i in {0,1}, j in {0,1}
        # B: j in {0,1}, k in {0}
        # C: i in {0,1}, k in {0}
        
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 1: A00 B00 -> C00
        c = col_start
        U[a_row_offset*m + (0+a_col_offset)][c] += 1
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += 1
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += 1
        col_start += 1
        
        # Term 2: A01 B10 -> C00
        c = col_start
        U[a_row_offset*m + (1+a_col_offset)][c] += 1 # A01 is i=0, j=1
        V[(1+b_row_offset)*p + (0+b_col_offset)][c] += 1
        W[(0+c_col_offset)*n + (0+a_row_offset)][c] += 1
        col_start += 1
        
        # Term 3: A10 B00 -> C10
        c = col_start
        U[a_row_offset*m + (1+a_col_offset)][c] += 1 # A10 is i=1, j=0
        V[(0+b_row_offset)*p + (0+b_col_offset)][c] += 1
        W[(0+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        # Term 4: A11 B10 -> C10
        c = col_start
        U[a_row_offset*m + (1+a_col_offset+1)][c] += 1 # A11 is i=1, j=1
        V[(1+b_row_offset)*p + (0+b_col_offset)][c] += 1
        W[(0+c_col_offset)*n + (1+a_row_offset)][c] += 1
        col_start += 1
        
        return col_start

    # Block 1: A0 B00 -> C0 (cols 0-1)
    col_offset = add_strassen_block(0, 0, 0, 0, 0, 0, col_offset)
    # Block 2: A1 B10 -> C0 (cols 0-1)
    col_offset = add_strassen_block(0, 2, 2, 0, 0, 0, col_offset)
    # Block 3: A0 B01 -> C1 (cols 2-3)
    col_offset = add_strassen_block(0, 0, 0, 2, 0, 2, col_offset)
    # Block 4: A1 B11 -> C1 (cols 2-3)
    col_offset = add_strassen_block(0, 2, 2, 2, 0, 2, col_offset)
    # Block 5: A0 B02 -> C2 (col 4)
    col_offset = add_schoolbook_block(0, 0, 0, 4, 0, 4, col_offset)
    # Block 6: A1 B12 -> C2 (col 4)
    col_offset = add_schoolbook_block(0, 2, 2, 4, 0, 4, col_offset)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
