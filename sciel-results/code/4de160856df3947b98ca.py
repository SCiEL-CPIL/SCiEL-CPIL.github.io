"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication via:
    - Level 1: Strassen on 2x2 grid of 3x3 blocks (7 terms)
    - Level 2: Schoolbook on each 3x3 product (27 scalar mults per term)
    Total rank = 7 * 27 = 189
    
    Interface: returns {'U': (36,189), 'V': (36,189), 'W': (36,189)}
    Convention: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r gives C[i,k]
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # 7 Strassen combinations at level 1
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # where each block is 3x3
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    
    # Strassen terms as (left_combo, right_combo):
    # Each combo is a dict from (br,bc) to coefficient
    strassen_terms = [
        # T1: (A00+A11)*(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T2: (A10+A11)*B00
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # T3: A00*(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # T4: A11*(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # T5: (A00+A01)*B11
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # T6: (A10-A00)*(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # T7: (A01-A11)*(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Result block contributions: C_block = sum of +/- Tj
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],   # T1, T4, -T5, T7
        (0,1): [(2, 1), (4, 1)],                     # T3, T5
        (1,0): [(1, 1), (3, 1)],                     # T2, T4
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # T1, -T2, T3, T6
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, compute X*Y where X and Y are 3x3
        # linear combinations of the original 3x3 blocks
        # Decompose into schoolbook: for each (i, a, k) in 0..2
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # x_{i,a} = sum over (br,bc) in left_combo: coeff * A[3*br+i, 3*bc+a]
                    # y_{a,k} = sum over (br,bc) in right_combo: coeff * B[3*br+a, 3*bc+k]
                    
                    # U: coefficients of A entries
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V: coefficients of B entries
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W: contributes to result blocks with certain signs
                    for (rbr, rbc), contrib_list in result_contribs.items():
                        for (si, sign) in contrib_list:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                # W row index = k*n + i = k_total * 6 + i_total
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
