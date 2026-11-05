"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 tensor decomposition for 6x6 matrix multiplication
    using one level of Strassen (7 multiplications of 3x3 matrices) combined
    with schoolbook (27 scalar multiplications) for each 3x3 multiplication.
    
    Total rank = 7 * 27 = 189.
    
    The decomposition follows the convention:
    - U has shape (36, 189): U[i*6+j, r] gives coefficient of A[i,j] in rank-r term
    - V has shape (36, 189): V[j*6+k, r] gives coefficient of B[j,k] in rank-r term  
    - W has shape (36, 189): W[k*6+i, r] gives coefficient for C[i,k] in rank-r term
    
    The reconstruction is: C[i,k] = sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
    summed over j as well (this is the full tensor contraction).
    """
    n = m = p = 6
    rank = 189
    
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen decomposition for 6x6 = 2x2 blocks of 3x3
    # A = [[A00, A01],    B = [[B00, B01],    C = [[C00, C01],
    #       [A10, A11]]         [B10, B11]]         [C10, C11]]
    #
    # Each block is 3x3. Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    #
    # Strassen terms:
    # T1 = (A00 + A11) * (B00 + B11)
    # T2 = (A10 + A11) * B00
    # T3 = A00 * (B01 - B11)
    # T4 = A11 * (B10 - B00)
    # T5 = (A00 + A01) * B11
    # T6 = (A10 - A00) * (B00 + B01)
    # T7 = (A01 - A11) * (B10 + B11)
    #
    # Result:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each Strassen term, define the linear combinations
    # left_combo: dict from (block_row, block_col) to coefficient for A
    # right_combo: dict from (block_row, block_col) to coefficient for B
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
    
    # Result block contributions: which Strassen terms contribute to which output block, with signs
    # C_block = sum of (sign * T_term)
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00 = T1 + T4 - T5 + T7
        (0,1): [(2, 1), (4, 1)],                     # C01 = T3 + T5
        (1,0): [(1, 1), (3, 1)],                     # C10 = T2 + T4
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11 = T1 - T2 + T3 + T6
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        # For this Strassen term, we compute X * Y where:
        # X[i,a] = sum_{(br,bc) in left_combo} coeff * A[3*br+i, 3*bc+a]
        # Y[a,k] = sum_{(br,bc) in right_combo} coeff * B[3*br+a, 3*bc+k]
        # Z[i,k] = sum_a X[i,a] * Y[a,k]
        #
        # Schoolbook: for each (i, a, k), one scalar multiplication
        # contributes to Z[i,k] = sum_a X[i,a] * Y[a,k]
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # X[i,a] = sum over left_combo
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # Y[a,k] = sum over right_combo
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W: this term contributes to C at positions determined by result blocks
                    # For each output block (rbr, rbc) that this Strassen term contributes to:
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                # W row = k_total * n + i_total (trace convention)
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
