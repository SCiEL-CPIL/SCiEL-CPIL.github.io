"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 84 decomposition using block Strassen
    # 3 blocks of size 2 for p.
    # Each block is 4x4x2.
    # 4x4x2 is decomposed into 4 instances of 2x2x2.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 3 * 4 * 7 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(offset, A_row, A_col, B_row, B_col, C_row, C_col):
        # A_row, A_col: top-left index of A submatrix (0 or 1)
        # B_row, B_col: top-left index of B submatrix (0 or 1 for row, 0, 2, 4 for col)
        # C_row, C_col: top-left index of C submatrix
        
        # m1 = (x00+x11)(y00+y11)
        col = offset + 0
        U[(A_row + 0) * 4 + (A_col + 0)][col] = 1
        U[(A_row + 1) * 4 + (A_col + 1)][col] = 1
        V[(B_row + 0) * 6 + (B_col + 0)][col] = 1
        V[(B_row + 1) * 6 + (B_col + 1)][col] = 1
        W[(C_col + 0) * 4 + (C_row + 0)][col] = 1
        W[(C_col + 1) * 4 + (C_row + 1)][col] = 1
        
        # m2 = (x10+x11)y00
        col = offset + 1
        U[(A_row + 1) * 4 + (A_col + 0)][col] = 1
        U[(A_row + 1) * 4 + (A_col + 1)][col] = 1
        V[(B_row + 0) * 6 + (B_col + 0)][col] = 1
        W[(C_col + 0) * 4 + (C_row + 1)][col] = 1
        W[(C_col + 1) * 4 + (C_row + 1)][col] = -1
        
        # m3 = x00(y01-y11)
        col = offset + 2
        U[(A_row + 0) * 4 + (A_col + 0)][col] = 1
        V[(B_row + 0) * 6 + (B_col + 1)][col] = 1
        V[(B_row + 1) * 6 + (B_col + 1)][col] = -1
        W[(C_col + 1) * 4 + (C_row + 0)][col] = 1
        W[(C_col + 1) * 4 + (C_row + 1)][col] = 1
        
        # m4 = x11(y10-y00)
        col = offset + 3
        U[(A_row + 1) * 4 + (A_col + 1)][col] = 1
        V[(B_row + 1) * 6 + (B_col + 0)][col] = 1
        V[(B_row + 0) * 6 + (B_col + 0)][col] = -1
        W[(C_col + 0) * 4 + (C_row + 0)][col] = 1
        W[(C_col + 0) * 4 + (C_row + 1)][col] = 1
        
        # m5 = (x00+x01)y11
        col = offset + 4
        U[(A_row + 0) * 4 + (A_col + 0)][col] = 1
        U[(A_row + 0) * 4 + (A_col + 1)][col] = 1
        V[(B_row + 1) * 6 + (B_col + 1)][col] = 1
        W[(C_col + 0) * 4 + (C_row + 0)][col] = -1
        W[(C_col + 1) * 4 + (C_row + 0)][col] = 1
        
        # m6 = (x10-x00)(y00+y01)
        col = offset + 5
        U[(A_row + 1) * 4 + (A_col + 0)][col] = 1
        U[(A_row + 0) * 4 + (A_col + 0)][col] = -1
        V[(B_row + 0) * 6 + (B_col + 0)][col] = 1
        V[(B_row + 0) * 6 + (B_col + 1)][col] = 1
        W[(C_col + 1) * 4 + (C_row + 1)][col] = 1
        
        # m7 = (x01-x11)(y10+y11)
        col = offset + 6
        U[(A_row + 0) * 4 + (A_col + 1)][col] = 1
        U[(A_row + 1) * 4 + (A_col + 1)][col] = -1
        V[(B_row + 1) * 6 + (B_col + 0)][col] = 1
        V[(B_row + 1) * 6 + (B_col + 1)][col] = 1
        W[(C_col + 0) * 4 + (C_row + 0)][col] = 1

    offset = 0
    for b in range(3):
        b_col = 2 * b
        # 4 instances of 2x2x2 for this block
        # 1. A[0:2, 0:2] * B[0:2, b_col:b_col+2] -> C[0:2, b_col:b_col+2]
        add_strassen(offset, 0, 0, 0, b_col, 0, b_col)
        offset += 7
        
        # 2. A[0:2, 2:4] * B[2:4, b_col:b_col+2] -> C[0:2, b_col:b_col+2]
        add_strassen(offset, 0, 1, 1, b_col, 0, b_col)
        offset += 7
        
        # 3. A[2:4, 0:2] * B[0:2, b_col:b_col+2] -> C[2:4, b_col:b_col+2]
        add_strassen(offset, 1, 0, 0, b_col, 1, b_col)
        offset += 7
        
        # 4. A[2:4, 2:4] * B[2:4, b_col:b_col+2] -> C[2:4, b_col:b_col+2]
        add_strassen(offset, 1, 1, 1, b_col, 1, b_col)
        offset += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
