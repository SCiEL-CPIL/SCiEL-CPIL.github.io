"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: 2-level recursive Strassen.
    Level 1: Split 6x6 into 2x2 blocks of 3x3. Use Strassen (7 terms).
    Level 2: For each 3x3 sub-multiplication, use schoolbook (27 terms).
    Total rank = 7 * 27 = 189.
    
    The factor matrices U, V, W have shapes:
      U: (36, 189) - linear combinations of A entries
      V: (36, 189) - linear combinations of B entries
      W: (36, 189) - coefficients for C entries
    
    Convention: C[i,k] = sum_{j,r} A[i,j] * B[j,k]
    U row index = i*6 + j (for A[i,j])
    V row index = j*6 + k (for B[j,k])
    W row index = k*6 + i (for C[i,k], following W[k*n+i, r] convention)
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Level 1: Strassen on 2x2 blocks of 3x3
    # A = [[A00, A01],   B = [[B00, B01],   C = [[C00, C01],
    #       [A10, A11]]       [B10, B11]]       [C10, C11]]
    # Each block is 3x3.
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    
    # 7 Strassen terms, each computing a 3x3 = 3x3 product:
    # T0: (A00+A11) * (B00+B11)
    # T1: (A10+A11) * B00
    # T2: A00 * (B01-B11)
    # T3: A11 * (B10-B00)
    # T4: (A00+A01) * B11
    # T5: (A10-A00) * (B00+B01)
    # T6: (A01-A11) * (B10+B11)
    
    # Each term is defined by:
    #   left: dict {(br,bc): coeff} for linear combo of A blocks
    #   right: dict {(br,bc): coeff} for linear combo of B blocks
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
    
    # Result block contributions: C_block = sum of +/- Tj
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
        # This Strassen term computes X * Y where:
        #   X[i,a] = sum_{(br,bc) in left_combo} coeff * A[3*br+i, 3*bc+a]
        #   Y[a,k] = sum_{(br,bc) in right_combo} coeff * B[3*br+a, 3*bc+k]
        #
        # Product Z[i,k] = sum_a X[i,a] * Y[a,k]
        #
        # Schoolbook decomposition: for each (i, a, k) in 0..2, one scalar mult
        #   x_ia * y_ak
        # contributes to Z[i,k].
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # Fill U: coefficient of A[row_A, col_A] in x_ia
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # Fill V: coefficient of B[row_B, col_B] in y_ak
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # Fill W: this term contributes to C[i_total, k_total]
                    # where i_total = 3*rbr + i, k_total = 3*rbc + k
                    # W row index = k_total * 6 + i_total
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
