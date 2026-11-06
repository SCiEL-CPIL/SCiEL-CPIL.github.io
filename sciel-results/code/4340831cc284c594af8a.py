"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: One level of Strassen on 6x6 = 2x2 blocks of 3x3.
    - 7 Strassen multiplications of 3x3 matrices
    - Each 3x3 multiplication decomposed via schoolbook (rank 27)
    - Total rank = 7 * 27 = 189
    
    Interface:
      U: shape (n*m, rank) = (36, 189), entries for input A
      V: shape (m*p, rank) = (36, 189), entries for input B  
      W: shape (p*n, rank) = (36, 189), entries for output C
      Convention: W[k*n + i, r] corresponds to C[i,k]
      U[i*m + j, r] corresponds to A[i,j]
      V[j*p + k, r] corresponds to B[j,k]
      C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # 7 Strassen combinations at level 1
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # where each block is 3x3
    # A00 = rows 0-2, cols 0-2; A01 = rows 0-2, cols 3-5
    # A10 = rows 3-5, cols 0-2; A11 = rows 3-5, cols 3-5
    
    # Strassen terms (each is a 3x3 = 3x3 multiplication):
    # T1: (A00+A11)*(B00+B11)
    # T2: (A10+A11)*B00
    # T3: A00*(B01-B11)
    # T4: A11*(B10-B00)
    # T5: (A00+A01)*B11
    # T6: (A10-A00)*(B00+B01)
    # T7: (A01-A11)*(B10+B11)
    
    # Result blocks:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
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
        # For this Strassen term, we need to compute X*Y where
        # X_{i,a} = sum over blocks in left_combo: coeff * A[block_row, block_col]_{i,a}
        # Y_{a,k} = sum over blocks in right_combo: coeff * B[block_row, block_col]_{a,k}
        #
        # X is 3x3, Y is 3x3. The product Z = X*Y is 3x3.
        # Z_{i,k} = sum_a X_{i,a} * Y_{a,k}
        #
        # We decompose this into scalar multiplications using schoolbook.
        # For each (i, a, k) in 0..2, we have one scalar multiplication:
        #   x_{i,a} * y_{a,k}
        # where x_{i,a} is a linear combination of entries of A,
        # and y_{a,k} is a linear combination of entries of B.
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # x_{i,a} = sum over (br,bc) in left_combo: coeff * A_{3*br+i, 3*bc+a}
                    # y_{a,k} = sum over (br,bc) in right_combo: coeff * B_{3*br+a, 3*bc+k}
                    
                    # For U: entry at row (3*br+i)*6 + (3*bc+a) for each block in left_combo
                    # For V: entry at row (3*br+a)*6 + (3*bc+k) for each block in right_combo
                    # For W: this contributes to C at position determined by result blocks
                    
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W: this term contributes to the result.
                    # The Strassen term T_{t_idx} contributes to result blocks
                    # with certain signs. For result block (rbr, rbc), the
                    # entry at local position (i, k) gets the sign.
                    # C_{rbr, rbc}[i, k] = sum over contributions
                    # C is 6x6, so row = 3*rbr + i, col = 3*rbc + k
                    # W index: W[k*n + i, col] per the trace convention W[k*n+i,r]
                    # So W row index = k_total*n + i_total, meaning for output C[i_total, k_total],
                    # the row in W is k_total*n + i_total.
                    
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
