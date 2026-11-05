"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: One level of Strassen on 6x6 split into 2x2 blocks of 3x3,
    giving 7 Strassen terms. Each term is a 3x3 matrix multiplication,
    decomposed via schoolbook (rank 27). Total rank = 7 * 27 = 189.
    
    The decomposition follows the trace convention:
      C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where n=m=p=6.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen terms at level 1:
    # A = [[A00, A01],    B = [[B00, B01],    C = [[C00, C01],
    #       [A10, A11]]         [B10, B11]]         [C10, C11]]
    # Each block is 3x3.
    #
    # T1 = (A00+A11)*(B00+B11)
    # T2 = (A10+A11)*B00
    # T3 = A00*(B01-B11)
    # T4 = A11*(B10-B00)
    # T5 = (A00+A01)*B11
    # T6 = (A10-A00)*(B00+B01)
    # T7 = (A01-A11)*(B10+B11)
    #
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each term, define the linear combinations for left (A-side) and right (B-side) factors.
    # Block (br, bc) means rows 3*br..3*br+2, cols 3*bc..3*bc+2.
    strassen_terms = [
        # (left_combo, right_combo) where each is dict {(br,bc): coeff}
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),   # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}),              # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),             # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),             # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}),              # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),    # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),    # T7
    ]
    
    # Result block contributions: which terms contribute to which result block with what sign
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],    # C00 = T1 + T4 - T5 + T7
        (0,1): [(2, 1), (4, 1)],                      # C01 = T3 + T5
        (1,0): [(1, 1), (3, 1)],                      # C10 = T2 + T4
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],    # C11 = T1 - T2 + T3 + T6
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, compute X * Y where:
        #   X[i,a] = sum_{(br,bc) in left_combo} coeff_l * A[3*br+i, 3*bc+a]
        #   Y[a,k] = sum_{(br,bc) in right_combo} coeff_r * B[3*br+a, 3*bc+k]
        #   Z[i,k] = sum_a X[i,a] * Y[a,k]
        #
        # Schoolbook: for each (i, a, k) in 0..2, one scalar multiplication X[i,a]*Y[a,k]
        # contributes to Z[i,k].
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U entries: for each block in left_combo, A[3*br+i, 3*bc+a]
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V entries: for each block in right_combo, B[3*br+a, 3*bc+k]
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W entries: this term contributes to result blocks with certain signs.
                    # The local (i,k) position in the 3x3 result block maps to global
                    # C[3*rbr+i, 3*rbc+k]. W row = k_total * n + i_total.
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
