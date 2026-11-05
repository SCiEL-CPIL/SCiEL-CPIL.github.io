"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: 2-level recursive Strassen.
    Level 1: Partition 6x6 into 2x2 blocks of 3x3 -> 7 multiplications of 3x3 matrices.
    Level 2: Each 3x3 multiplication is decomposed via schoolbook (27 scalar multiplications).
    Total rank = 7 * 27 = 189.
    
    Interface:
      U: shape (36, 189) - factors for input A (6x6)
      V: shape (36, 189) - factors for input B (6x6)
      W: shape (36, 189) - factors for output C (6x6), following W[k*n+i, r] convention
    
    The decomposition satisfies:
      C[i,k] = sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]  (summed over j)
    
    All entries are integers.
    """
    n = m = p = 6
    rank = 189
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Level 1: Strassen decomposition of 6x6 into 2x2 blocks of 3x3
    # A = [[A00, A01],   B = [[B00, B01],   C = [[C00, C01],
    #       [A10, A11]]       [B10, B11]]       [C10, C11]]
    #
    # 7 Strassen terms (each is a 3x3 matrix product):
    # T1: (A00 + A11) * (B00 + B11)
    # T2: (A10 + A11) * B00
    # T3: A00 * (B01 - B11)
    # T4: A11 * (B10 - B00)
    # T5: (A00 + A01) * B11
    # T6: (A10 - A00) * (B00 + B01)
    # T7: (A01 - A11) * (B10 + B11)
    #
    # Result blocks:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each Strassen term, define the linear combinations for left and right factors
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    strassen_terms = [
        # T1: (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # T2: (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # T3: A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # T4: A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # T5: (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # T6: (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # T7: (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Result block contributions: C_block = sum of signed Strassen terms
    result_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00 = T1 + T4 - T5 + T7
        (0, 1): [(2, 1), (4, 1)],                     # C01 = T3 + T5
        (1, 0): [(1, 1), (3, 1)],                     # C10 = T2 + T4
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11 = T1 - T2 + T3 + T6
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, compute X * Y where:
        # X_{i,a} = sum over (br,bc) in left_combo: coeff * A_{3*br+i, 3*bc+a}
        # Y_{a,k} = sum over (br,bc) in right_combo: coeff * B_{3*br+a, 3*bc+k}
        #
        # The product Z = X*Y has Z_{i,k} = sum_a X_{i,a} * Y_{a,k}
        # We decompose into schoolbook: for each (i, a, k), one scalar multiplication.
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U entries: for each block in left_combo, add coefficient to U at (row_A, col_A)
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V entries: for each block in right_combo, add coefficient to V at (row_B, col_B)
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W entries: this term contributes to output C with sign from result_contribs
                    # Output position: i_total = 3*rbr + i, k_total = 3*rbc + k
                    # W row index = k_total * n + i_total = k_total * 6 + i_total
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
