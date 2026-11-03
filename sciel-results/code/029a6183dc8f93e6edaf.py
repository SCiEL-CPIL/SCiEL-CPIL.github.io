import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    rank = 28

    # Initialize U, V, W as lists of lists of floats
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen decomposition for 2x2x2
    # Let A, B, C be 2x2 matrices.
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> global indices depend on block position
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # The 7 Strassen products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction of C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Define the coefficients for U, V, W for each of the 7 products
    # U part: coefficients for A00, A01, A10, A11
    # V part: coefficients for B00, B01, B10, B11
    # W part: coefficients for C00, C01, C10, C11 (but note W index is k*n+i)
    
    # Local indices for 2x2 block:
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    # U coefficients (for A)
    # M1: A00(1), A11(1) -> {0: 1, 3: 1}
    # M2: A10(1), A11(1) -> {2: 1, 3: 1}
    # M3: A00(1) -> {0: 1}
    # M4: A11(1) -> {3: 1}
    # M5: A00(1), A01(1) -> {0: 1, 1: 1}
    # M6: A10(1), A00(-1) -> {2: 1, 0: -1}
    # M7: A01(1), A11(-1) -> {1: 1, 3: -1}
    
    u_coeffs = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: 1.0},
        {0: 1.0},
        {3: 1.0},
        {0: 1.0, 1: 1.0},
        {2: 1.0, 0: -1.0},
        {1: 1.0, 3: -1.0}
    ]
    
    # V coefficients (for B)
    # M1: B00(1), B11(1) -> {0: 1, 3: 1}
    # M2: B00(1) -> {0: 1}
    # M3: B01(1), B11(-1) -> {1: 1, 3: -1}
    # M4: B10(1), B00(-1) -> {2: 1, 0: -1}
    # M5: B11(1) -> {3: 1}
    # M6: B00(1), B01(1) -> {0: 1, 1: 1}
    # M7: B10(1), B11(1) -> {2: 1, 3: 1}
    
    v_coeffs = [
        {0: 1.0, 3: 1.0},
        {0: 1.0},
        {1: 1.0, 3: -1.0},
        {2: 1.0, 0: -1.0},
        {3: 1.0},
        {0: 1.0, 1: 1.0},
        {2: 1.0, 3: 1.0}
    ]
    
    # W coefficients (for C)
    # Note: W index is k*n + i.
    # For a 2x2 block of C, local indices:
    # 0: C00 (k=0, i=0)
    # 1: C01 (k=1, i=0)
    # 2: C10 (k=0, i=1)
    # 3: C11 (k=1, i=1)
    # Wait, the convention is W[k*n+i].
    # If we are placing a 2x2 block at (row_offset, col_offset) in the global C matrix:
    # Global C row i_g = row_offset + i_l
    # Global C col k_g = col_offset + k_l
    # W index = k_g * n + i_g
    
    # Reconstruction:
    # M1: C00(1), C11(-1)? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W contributions:
    # M1: C00(1), C11(1) -> Local {0: 1, 3: 1}
    # M2: C10(1), C11(-1) -> Local {2: 1, 3: -1}
    # M3: C01(1), C11(1) -> Local {1: 1, 3: 1}
    # M4: C00(1), C10(1) -> Local {0: 1, 2: 1}
    # M5: C00(-1), C01(1) -> Local {0: -1, 1: 1}
    # M6: C11(1) -> Local {3: 1}
    # M7: C00(1) -> Local {0: 1}
    
    w_coeffs = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0}
    ]

    # We have 4 blocks of 2x2:
    # Block 0: A[:2, :2] * B[:2, :2] -> C[:2, :2]
    # Block 1: A[:2, 2:] * B[:2, 2:] -> C[:2, 2:]
    # Block 2: A[2:, :2] * B[2:, :2] -> C[2:, :2]  (Wait, A is 2x4, so A has 2 rows. A[2:] is empty? No, A is 2 rows, 4 cols.
    # Let's re-verify the block mapping.
    # A (2x4) is split into A_L (2x2) and A_R (2x2).
    # B (4x4) is split into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2).
    # C (2x4) is split into C_L (2x2) and C_R (2x2).
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # So we have 4 Strassen multiplications:
    # 1. A_L * B_TL -> contributes to C_L
    # 2. A_R * B_BL -> contributes to C_L
    # 3. A_L * B_TR -> contributes to C_R
    # 4. A_R * B_BR -> contributes to C_R

    def place_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, rank_offset):
        """
        Adds the Strassen decomposition for a 2x2 * 2x2 block.
        a_col_offset: column offset in A (0 or 2)
        b_row_offset: row offset in B (0 or 2)
        b_col_offset: column offset in B (0 or 2)
        c_row_offset: row offset in C (always 0 for 2x4 output? No, C is 2x4. Rows are 0,1.
                      Wait, C is n x p = 2 x 4.
                      The blocks of C are 2x2.
                      C_L is columns 0,1. C_R is columns 2,3.
                      Rows are always 0,1.
                      So c_row_offset is always 0?
                      Yes, because n=2, and the blocks are 2x2.
                      So c_row_offset is 0 for both C_L and C_R.
        """
        for s in range(7):
            col = rank_offset + s
            
            # U part (A)
            for local_idx, coeff in u_coeffs[s].items():
                # local_idx 0->00, 1->01, 2->10, 3->11
                i_l = local_idx // 2
                j_l = local_idx % 2
                i_g = i_l # c_row_offset is 0 for A rows? A rows are 0,1.
                j_g = a_col_offset + j_l
                U[i_g * m + j_g][col] += coeff

            # V part (B)
            for local_idx, coeff in v_coeffs[s].items():
                i_l = local_idx // 2
                j_l = local_idx % 2
                i_g = b_row_offset + i_l
                j_g = b_col_offset + j_l
                V[i_g * p + j_g][col] += coeff

            # W part (C)
            for local_idx, coeff in w_coeffs[s].items():
                # local_idx 0->C00, 1->C01, 2->C10, 3->C11
                i_l = local_idx % 2 # Row in 2x2 block
                k_l = local_idx // 2 # Col in 2x2 block
                i_g = c_row_offset + i_l
                k_g = c_col_offset + k_l
                W[k_g * n + i_g][col] += coeff

    # Block 1: A_L (cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    # a_col_offset=0, b_row_offset=0, b_col_offset=0, c_row_offset=0, c_col_offset=0
    place_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_R (cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    # a_col_offset=2, b_row_offset=2, b_col_offset=0, c_row_offset=0, c_col_offset=0
    place_strassen_block(2, 2, 0, 0, 0, 7)
    
    # Block 3: A_L (cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    # a_col_offset=0, b_row_offset=0, b_col_offset=2, c_row_offset=0, c_col_offset=2
    place_strassen_block(0, 0, 2, 0, 2, 14)
    
    # Block 4: A_R (cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    # a_col_offset=2, b_row_offset=2, b_col_offset=2, c_row_offset=0, c_col_offset=2
    place_strassen_block(2, 2, 2, 0, 2, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
