"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: 2-level recursive approach.
    Level 1: Partition 6x6 into 2x2 blocks of 3x3 -> 7 Strassen multiplications of 3x3.
    Level 2: Each 3x3 multiplication is decomposed via schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    
    The tensor decomposition satisfies:
      sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] = C[i,k]
    for the matrix product C = A * B.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Level 1 Strassen: 7 terms for 6x6 = 2x2 blocks of 3x3
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # Each block is 3x3, with block (br, bc) covering rows 3*br..3*br+2, cols 3*bc..3*bc+2
    
    # Strassen terms: Tj = Xj * Yj where Xj, Yj are 3x3 linear combos of blocks
    # (left_combo, right_combo): dict from (block_row, block_col) to coefficient
    strassen_terms = [
        # T0: (A00+A11)*(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T1: (A10+A11)*B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # T2: A00*(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # T3: A11*(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # T4: (A00+A01)*B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # T5: (A10-A00)*(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # T6: (A01-A11)*(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Result block contributions: C_block = sum of signed Tj
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, compute X*Y where:
        # X[i,a] = sum_{(br,bc) in left_combo} coeff * A[3*br+i, 3*bc+a]
        # Y[a,k] = sum_{(br,bc) in right_combo} coeff * B[3*br+a, 3*bc+k]
        # Z[i,k] = sum_a X[i,a] * Y[a,k]
        #
        # Schoolbook decomposition: for each (i, a, k), one scalar multiplication
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U entries: for each block in left_combo
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V entries: for each block in right_combo
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W entries: this term contributes to result blocks
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
