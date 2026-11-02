"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen multiplications
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+i)
    # M2 = (c+d)(e+i)
    # M3 = a(f-j)
    # M4 = c(g-i)
    # M5 = d(g-i)
    # M6 = (a+b)(h-j)
    # M7 = (b+c)(k-l)
    
    # We need to map the 2x2 indices to the 2x4 and 4x4 indices.
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    # B is 4x4. 
    # B_TL is rows 0,1 cols 0,1.
    # B_BL is rows 2,3 cols 0,1.
    # B_TR is rows 0,1 cols 2,3.
    # B_BR is rows 2,3 cols 2,3.
    
    # For each of the 4 blocks, we use 7 multiplications.
    # Block 0: A_L * B_TL -> C_TL (cols 0,1 of C)
    # Block 1: A_R * B_BL -> C_BL (cols 0,1 of C)
    # Block 2: A_L * B_TR -> C_TR (cols 2,3 of C)
    # Block 3: A_R * B_BR -> C_BR (cols 2,3 of C)
    
    # C_TL + C_BL = C_L (cols 0,1)
    # C_TR + C_BR = C_R (cols 2,3)
    
    # Indices in U: i*m + j. i in 0..1, j in 0..3.
    # Indices in V: j*p + k. j in 0..3, k in 0..3.
    # Indices in W: k*n + i. k in 0..3, i in 0..1.
    
    # Strassen 2x2 indices:
    # A: a(0,0), b(0,1), c(1,0), d(1,1)
    # B: e(0,0), f(0,1), g(1,0), h(1,1)
    # C: c(0,0)=M1-M4-M5, c(0,1)=M3+M5, c(1,0)=M2+M4, c(1,1)=M1-M2+M6
    
    # For each block, we assign 7 columns of U, V, W.
    
    def add_strassen_block(offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset, a_col_offset: start row/col in A for this 2x2 block
        # b_row_offset, b_col_offset: start row/col in B for this 2x2 block
        # c_row_offset, c_col_offset: start row/col in C for this 2x2 block
        
        # M1
        col = offset + 0
        # A part: a+d
        U[a_row_offset * m + a_col_offset][col] = 1
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        # B part: e+i
        V[b_row_offset * p + b_col_offset][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        # C part: c00, c11
        W[c_row_offset * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        
        # M2
        col = offset + 1
        # A part: c+d
        U[(a_row_offset+1) * m + a_col_offset][col] = 1
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        # B part: e+i
        V[b_row_offset * p + b_col_offset][col] = 1
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        # C part: c10, c11
        W[(c_row_offset+1) * n + c_col_offset][col] = 1
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = -1
        
        # M3
        col = offset + 2
        # A part: a
        U[a_row_offset * m + a_col_offset][col] = 1
        # B part: f-j
        V[b_row_offset * p + (b_col_offset+1)][col] = 1
        V[(b_row_offset+1) * p + b_col_offset][col] = -1
        # C part: c01, c00
        W[c_row_offset * n + (c_col_offset+1)][col] = 1
        W[c_row_offset * n + c_col_offset][col] = -1
        
        # M4
        col = offset + 3
        # A part: c
        U[(a_row_offset+1) * m + a_col_offset][col] = 1
        # B part: g-i
        V[(b_row_offset+1) * p + b_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset][col] = -1
        # C part: c10, c00
        W[(c_row_offset+1) * n + c_col_offset][col] = 1
        W[c_row_offset * n + c_col_offset][col] = -1
        
        # M5
        col = offset + 4
        # A part: d
        U[(a_row_offset+1) * m + (a_col_offset+1)][col] = 1
        # B part: g-i
        V[(b_row_offset+1) * p + b_col_offset][col] = 1
        V[b_row_offset * p + b_col_offset][col] = -1
        # C part: c01, c10
        W[c_row_offset * n + (c_col_offset+1)][col] = 1
        W[(c_row_offset+1) * n + c_col_offset][col] = -1
        
        # M6
        col = offset + 5
        # A part: a+b
        U[a_row_offset * m + a_col_offset][col] = 1
        U[a_row_offset * m + (a_col_offset+1)][col] = 1
        # B part: h-j
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        V[(b_row_offset+1) * p + b_col_offset][col] = -1
        # C part: c11
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        
        # M7
        col = offset + 6
        # A part: b+c
        U[a_row_offset * m + (a_col_offset+1)][col] = 1
        U[(a_row_offset+1) * m + a_col_offset][col] = 1
        # B part: k-l
        V[(b_row_offset+1) * p + (b_col_offset+1)][col] = 1
        V[b_row_offset * p + (b_col_offset+1)][col] = -1
        # C part: c11
        W[(c_row_offset+1) * n + (c_col_offset+1)][col] = 1
        
    # Block 0: A_L (rows 0-1, cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C (rows 0-1, cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A_R (rows 0-1, cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C (rows 0-1, cols 0-1)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (rows 0-1, cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C (rows 0-1, cols 2-3)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A_R (rows 0-1, cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C (rows 0-1, cols 2-3)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
