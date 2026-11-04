"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 3x3 grid of 2x2 blocks
    # Outer: 3x3 Schoolbook (Rank 27)
    # Inner: 2x2 Strassen (Rank 7)
    # Total Rank: 27 * 7 = 189
    
    rank_outer = 27
    rank_inner = 7
    total_rank = rank_outer * rank_inner
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # U_strassen[i, k]
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 1, 0, 0, -1], # A10
        [0, 0, 1, 0, 1, 0, 1],  # A01
        [1, 1, 0, 1, 0, 0, 0]   # A11
    ]
    
    # V_strassen[j, k]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 1, 1, 0],  # B10
        [1, 0, 0, 1, 0, 0, 1],  # B01
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W_strassen[k, i]
    # C00 = M1 + M4 - M5 - M7
    # C10 = M2 + M3
    # C01 = M5 + M6
    # C11 = M1 - M2 + M3 + M7
    W_s = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 1, 1, 0, 0, 0, 0],   # C10
        [0, 0, 0, 0, 1, 1, 0],   # C01
        [1, -1, 1, 0, 0, 0, 1]   # C11
    ]
    
    # Schoolbook 3x3 coefficients
    # Rank 27
    U_sb = [[0]*27 for _ in range(9)]
    V_sb = [[0]*27 for _ in range(9)]
    W_sb = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j][col] = 1
                V_sb[j*3+k][col] = 1
                W_sb[k*3+i][col] = 1
                col += 1
                
    # Combine
    for i1 in range(3):
        for i2 in range(2):
            row_idx = i1 * 2 + i2
            for k1 in range(rank_outer):
                if U_sb[i1*3 + (i2//1)][k1] != 0: # Wait, U_sb is for 3x3 blocks.
                    # The row index for U_sb is i1*3 + i2? No.
                    # U_sb rows are 0..8.
                    # i1 is block row (0..2), i2 is local row (0..1).
                    # The local row i2 corresponds to the row within the 2x2 block.
                    # But U_sb is for the 3x3 block multiplication.
                    # The indices of U_sb are (block_row, block_col).
                    # So U_sb[i1, j1] is the coefficient for A[i1, j1] * B[j1, k1].
                    # The local indices i2, j2, k2 are for the 2x2 blocks.
                    
                    pass

    # Correct Combination Logic:
    # U_6x6[row, rank] = U_3x3[i1, k1] * U_2x2[i2, k2]
    # row = i1 * 2 + i2
    # rank = k1 * 7 + k2
    
    for i1 in range(3):
        for i2 in range(2):
            row = i1 * 2 + i2
            for k1 in range(rank_outer):
                for k2 in range(rank_inner):
                    val = U_sb[i1*3 + 0][k1] # Wait, U_sb row index is i1*3 + j1?
                    # No, U_sb is 9x27.
                    # Row index of U_sb is i1*3 + j1? No.
                    # U_sb[i, k] where i is 0..8.
                    # i corresponds to (i1, j1) in the 3x3 block grid?
                    # No, U_sb is for the 3x3 matrix multiplication.
                    # The input to the 3x3 multiplication is a 3x3 matrix of blocks.
                    # So U_sb rows are indexed by (i1, j1).
                    # But U_6x6 rows are indexed by (i1, i2).
                    # This means we need to map (i1, i2) to the 3x3 block structure.
                    
                    # Actually, the tensor product is:
                    # T_6x6 = T_3x3 \otimes T_2x2
                    # U_6x6[(i1, i2), (k1, k2)] = U_3x3[i1, k1] * U_2x2[i2, k2]
                    # This is only valid if the indices align.
                    
                    pass

    # Let's just build it directly.
    for i1 in range(3):
        for i2 in range(2):
            row = i1 * 2 + i2
            for k1 in range(rank_outer):
                for k2 in range(rank_inner):
                    rank_idx = k1 * rank_inner + k2
                    # U_3x3 row is i1? No, U_3x3 has 9 rows.
                    # The row index of U_3x3 is i1*3 + j1?
                    # No, the row index of U_3x3 is the input index of the 3x3 matrix.
                    # The input of the 3x3 matrix is a 3x3 grid of 2x2 blocks.
                    # So the input index is (i1, j1).
                    # But U_6x6 row is (i1, i2).
                    # This implies that the "column" index of U_3x3 must be related to i2?
                    # No, the tensor product structure is:
                    # U_6x6[(i1, i2), (k1, k2)] = U_3x3[i1, k1] * U_2x2[i2, k2]
                    # This requires that the "column" index of U_3x3 is k1, and the "row" index is i1.
                    # But U_3x3 has 9 rows.
                    # The row index of U_3x3 is (i1, j1).
                    # So we need to sum over j1?
                    # No, the tensor product of decompositions is not simply element-wise product of U matrices.
                    
                    pass

    # Correct approach:
    # U_6x6[row, rank] is non-zero if there exists (i1, j1) such that...
    # Actually, the decomposition is:
    # C[i1, k1] = sum_j1 A[i1, j1] B[j1, k1]
    # Each A[i1, j1] is a 2x2 matrix.
    # So C[i1, k1] = sum_j1 (A_block[i1, j1] * B_block[j1, k1])
    # The product of two 2x2 matrices is done by Strassen.
    # So C_block[i1, k1] = sum_j1 Strassen(A_block[i1, j1], B_block[j1, k1])
    
    # This means the rank is 27 * 7.
    # The terms are indexed by (i1, j1, k1) for the 3x3 part, and (m1) for the Strassen part.
    
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                for m1 in range(7):
                    rank_idx = (i1 * 3 + j1) * 7 + m1 # This is not right.
                    # The 3x3 schoolbook has 27 terms.
                    # Term t is (i1, j1, k1).
                    # t = i1 * 9 + j1 * 3 + k1.
                    t = i1 * 9 + j1 * 3 + k1
                    rank_idx = t * 7 + m1
                    
                    # U_6x6[row, rank_idx]
                    # row = i1 * 2 + i2
                    # U_3x3[i1*3+j1, t] = 1
                    # U_2x2[i2, m1]
                    
                    for i2 in range(2):
                        row = i1 * 2 + i2
                        U[row][rank_idx] = U_s[i2][m1]
                        
                    for j2 in range(2):
                        col = j1 * 2 + j2
                        V[col][rank_idx] = V_s[j2][m1]
                        
                    for k2 in range(2):
                        w_row = k1 * 2 + k2
                        W[w_row][rank_idx] = W_s[m1][k2]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
