"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition: split into 2x2 blocks of 3x3 matrices
    # 7 Strassen steps, each with 27 schoolbook terms for 3x3 multiplication
    rank = 7 * 27  # 189
    
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Block indexing for 2x2 blocks of 3x3:
    # A blocks: A00 (rows 0-2, cols 0-2), A01 (rows 0-2, cols 3-5)
    #           A10 (rows 3-5, cols 0-2), A11 (rows 3-5, cols 3-5)
    # B blocks: B00 (rows 0-2, cols 0-2), B01 (rows 0-2, cols 3-5)
    #           B10 (rows 3-5, cols 0-2), B11 (rows 3-5, cols 3-5)
    # C blocks: C00 (rows 0-2, cols 0-2), C01 (rows 0-2, cols 3-5)
    #           C10 (rows 3-5, cols 0-2), C11 (rows 3-5, cols 3-5)
    
    # For each block, the local indices (i, j, k) range over 0, 1, 2
    # Global index for A[i_block][j_block] at local (r, c):
    #   row = i_block * 3 + r, col = j_block * 3 + c
    #   U index = row * m + col = (i_block*3+r) * 6 + (j_block*3+c)
    
    # Global index for B[j_block][k_block] at local (r, c):
    #   row = j_block * 3 + r, col = k_block * 3 + c
    #   V index = row * p + col = (j_block*3+r) * 6 + (k_block*3+c)
    
    # Global index for C[i_block][k_block] at local (r, c):
    #   row = i_block * 3 + r, col = k_block * 3 + c
    #   W index = col * n + row = (k_block*3+c) * 6 + (i_block*3+r)
    
    # Strassen's 7 steps:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C combination:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each step s (0-6), for each local (i, j, k) in 0,1,2:
    #   col = s * 27 + i * 9 + j * 3 + k
    
    # Step definitions: (A_blocks_with_signs, B_blocks_with_signs)
    # A_blocks: list of (block_idx, sign) where block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # B_blocks: list of (block_idx, sign) where block_idx: 0=B00, 1=B01, 2=B10, 3=B11
    
    steps = [
        # M1 = (A00+A11)*(B00+B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2 = (A10+A11)*B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        # M3 = A00*(B01-B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        # M4 = A11*(B10-B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        # M5 = (A00+A01)*B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M6 = (A10-A00)*(B00+B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        # M7 = (A01-A11)*(B10+B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ]),
    ]
    
    # C combination: for each step s, which C blocks get which sign
    # C_blocks: 0=C00, 1=C01, 2=C10, 3=C11
    c_combination = [
        # M1: C00(+1), C01(+1), C10(+1), C11(+1) -- NO, let me recheck
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # So for each M_s, the C blocks it contributes to:
        [ (0, 1), (1, 1), (2, 1), (3, 1) ],  # M1 -> C00, C01, C10, C11? NO!
        
        # Let me redo this. Each M_s is a full 3x3 matrix.
        # M1 contributes: C00 += M1, C11 += M1
        # M2 contributes: C10 += M2, C11 -= M2
        # M3 contributes: C01 += M3, C11 += M3
        # M4 contributes: C00 -= M4, C10 += M4
        # M5 contributes: C01 += M5
        # M6 contributes: C00 += M6
        # M7 contributes: C00 -= M7, C11 += M7
    ]
    
    # Correct C combination per step:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(-1), C10(+1)
    # M5: C01(+1)
    # M6: C00(+1)
    # M7: C00(-1), C11(+1)
    
    c_combination = [
        [ (0, 1), (3, 1) ],  # M1
        [ (2, 1), (3, -1) ],  # M2
        [ (1, 1), (3, 1) ],  # M3
        [ (0, -1), (2, 1) ],  # M4
        [ (1, 1) ],  # M5
        [ (0, 1) ],  # M6
        [ (0, -1), (3, 1) ],  # M7
    ]
    
    for s in range(7):
        a_blocks, b_blocks = steps[s]
        c_blocks = c_combination[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    # Set U entries: for each A block with sign
                    for (ab, sign) in a_blocks:
                        # A block ab: row_block = ab // 2, col_block = ab % 2
                        row_block = ab // 2
                        col_block = ab % 2
                        # Local (i, j) in the 3x3
                        u_row = (row_block * 3 + i) * m + (col_block * 3 + j)
                        U[u_row][col] += sign
                    
                    # Set V entries: for each B block with sign
                    for (bb, sign) in b_blocks:
                        # B block bb: row_block = bb // 2, col_block = bb % 2
                        row_block = bb // 2
                        col_block = bb % 2
                        # Local (j, k) in the 3x3
                        v_row = (row_block * 3 + j) * p + (col_block * 3 + k)
                        V[v_row][col] += sign
                    
                    # Set W entries: for each C block with sign
                    for (cb, sign) in c_blocks:
                        # C block cb: row_block = cb // 2, col_block = cb % 2
                        row_block = cb // 2
                        col_block = cb % 2
                        # Local (i, k) in the 3x3
                        # W[k*n + i] where i is C row, k is C col
                        w_row = (col_block * 3 + k) * n + (row_block * 3 + i)
                        W[w_row][col] += sign
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
