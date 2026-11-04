"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    rank = 92
    U = [[0] * rank for _ in range(n * m)]  # 20 x 92
    V = [[0] * rank for _ in range(m * p)]  # 25 x 92
    W = [[0] * rank for _ in range(p * n)]  # 20 x 92
    
    def set_entry(mat, row, col, val):
        if 0 <= row < len(mat) and 0 <= col < len(mat[0]):
            mat[row][col] = val
    
    # Strassen decomposition for 4x4x4 sub-block
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bi is 2x2
    # C = [[C11, C12], [C21, C22]] where each Ci is 2x2
    #
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each 2x2x2 multiplication uses schoolbook (rank 8)
    # Total: 7 * 8 = 56 for the 4x4x4 part
    
    col = 0
    
    # Helper to add schoolbook 2x2x2 decomposition
    def add_schoolbook_2x2(a_block, b_block, c_block, col_offset):
        """Add schoolbook decomposition for 2x2x2 multiplication.
        a_block: (row_start, col_start) for 2x2 block in A
        b_block: (row_start, col_start) for 2x2 block in B  
        c_block: (row_start, col_start) for 2x2 block in C
        col_offset: starting column index
        Returns: new column index
        """
        nonlocal col
        ar, ac = a_block
        br, bc = b_block
        cr, cc = c_block
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A[i,j] * B[j,k] -> C[i,k]
                    a_row = ar + i
                    a_col = ac + j
                    b_row = br + j
                    b_col = bc + k
                    c_row = cr + k
                    c_col = cc + i
                    
                    if col_offset + (i*2 + j)*2 + k < rank:
                        set_entry(U, a_row * m + a_col, col_offset + (i*2 + j)*2 + k, 1)
                        set_entry(V, b_row * p + b_col, col_offset + (i*2 + j)*2 + k, 1)
                        set_entry(W, c_row * n + c_col, col_offset + (i*2 + j)*2 + k, 1)
        
        return col_offset + 8
    
    # Define block positions for 4x4 A, 4x4 B, 4x4 C
    # A is 4x4: blocks A11=(0,0), A12=(0,2), A21=(2,0), A22=(2,2)
    # B is 4x4: blocks B11=(0,0), B12=(0,2), B21=(2,0), B22=(2,2)
    # C is 4x4: blocks C11=(0,0), C12=(0,2), C21=(2,0), C22=(2,2)
    
    # But we need to handle the sums/differences in Strassen
    # For M1 = (A11+A22)(B11+B22):
    # This is NOT a simple schoolbook - it's a product of sums
    # We need to expand: (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each of these is a 2x2x2 multiplication
    
    # M1 = (A11+A22)(B11+B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22
    m1_terms = [
        ((0,0), (0,0), 1),   # A11*B11
        ((0,0), (2,0), 1),   # A11*B22 - wait, B22 is at (2,2) not (2,0)
    ]
    
    # Let me be more careful with block positions
    # A blocks: A11 rows 0-1, cols 0-1; A12 rows 0-1, cols 2-3; A21 rows 2-3, cols 0-1; A22 rows 2-3, cols 2-3
    # B blocks: B11 rows 0-1, cols 0-1; B12 rows 0-1, cols 2-3; B21 rows 2-3, cols 0-1; B22 rows 2-3, cols 2-3
    # C blocks: C11 rows 0-1, cols 0-1; C12 rows 0-1, cols 2-3; C21 rows 2-3, cols 0-1; C22 rows 2-3, cols 2-3
    
    # For the 4x4x4 part, all indices are within 0-3
    # U index: i*m + j where i is row in A (0-3), j is col in A (0-3)
    # V index: j*p + k where j is row in B (0-3), k is col in B (0-3)
    # W index: k*n + i where k is col in B (0-3), i is row in A (0-3)
    
    # M1 = (A11 + A22)(B11 + B22)
    # Expand: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # A11*B11: i in 0-1, j in 0-1 for A; j in 0-1, k in 0-1 for B; k in 0-1, i in 0-1 for C
    # A11*B22: i in 0-1, j in 0-1 for A; j in 2-3, k in 2-3 for B; k in 2-3, i in 0-1 for C
    # A22*B11: i in 2-3, j in 2-3 for A; j in 0-1, k in 0-1 for B; k in 0-1, i in 2-3 for C
    # A22*B22: i in 2-3, j in 2-3 for A; j in 2-3, k in 2-3 for B; k in 2-3, i in 2-3 for C
    
    # Each term is a 2x2x2 schoolbook with 8 terms
    # So M1 contributes 32 terms total? No, that's not right.
    
    # Actually, Strassen's algorithm works at the algorithmic level, not the tensor level.
    # The tensor rank reduction comes from the fact that the 7 products M1-M7 can be
    # computed with 7 scalar multiplications when the inputs are scalars.
    
    # For a proper tensor decomposition, I need to think about this differently.
    # The key insight is that the 4x4x4 tensor has rank at most 56 using Strassen,
    # compared to 64 for schoolbook.
    
    # Let me use a simpler approach: just use schoolbook for the entire 4x5x5
    # but with a smarter block decomposition.
    
    # Actually, the proven result from the prompt is that using Strassen for 2x2x2
    # sub-blocks reduces rank from 100 to 92. Let me implement this correctly.
    
    # The 4x5x5 can be decomposed as:
    # - 4x4x4 part (using Strassen, rank 56)
    # - 4x1x1 part (schoolbook, rank 4)
    # - 4x4x1 part (schoolbook, rank 16)
    # - 1x4x1 part (schoolbook, rank 4)
    # - 1x1x1 part (schoolbook, rank 1)
    # Total: 56 + 4 + 16 + 4 + 1 = 81? That doesn't match 92.
    
    # Let me reconsider. The dimensions are 4x5 and 5x5.
    # A is 4x5, B is 5x5, C = A*B is 4x5.
    # 
    # Partition:
    # A = [A1 A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 5x4, B2 is 5x1
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x4 -> 4x4, rank 64 schoolbook or 56 with Strassen
    # A2*B2: 4x1 * 1x4 -> 4x4, rank 16
    # Total: 56 + 16 = 72? Still not 92.
    
    # I think I'm overcomplicating this. Let me just implement the schoolbook
    # solution with rank 100, which is known to be correct.
    
    # Reset
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
