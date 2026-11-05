"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Apply 2-level recursive Strassen:
    # Level 1: partition 6x6 into 2x2 blocks of 3x3 -> 7 multiplications of 3x3
    # Level 2: partition each 3x3 into 2x2 blocks (sizes 2,1) -> 7 multiplications
    #   of sub-blocks of sizes (2,1)x(2,1) -> schoolbook for those
    # Total rank = 7 * 7 * 4 = 196  (each of the 7 level-1 Strassen terms
    # expands into 7 level-2 Strassen terms, each of which is a schoolbook
    # multiplication of a 2x2 by 2x2 matrix = 4 scalar multiplications)
    
    # We build the full 216x196 factor matrices U, V, W with integer entries.
    # U: shape (36, 196), V: shape (36, 196), W: shape (36, 196)
    
    n = m = p = 6
    rank = 196
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen for block sizes: at level 1, blocks are 3x3 (rows/cols 0-2, 3-5)
    # At level 2, each 3x3 is split into 2x2 blocks of sizes (2,1)
    #   top-left: rows 0-1, cols 0-1 (2x2)
    #   top-right: rows 0-1, col 2 (2x1)
    #   bottom-left: row 2, cols 0-1 (1x2)
    #   bottom-right: row 2, col 2 (1x1)
    
    # Level 2 Strassen for a 3x3 matrix product C = A * B where:
    #   A = [[A00, A01],    B = [[B00, B01],
    #         [A10, A11]]       [B10, B11]]
    # with A00,B00,C00 being 2x2; A01,B01,C01 being 2x1;
    #       A10,B10,C10 being 1x2; A11,B11,C11 being 1x1
    #
    # Strassen terms:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)B00
    #   M3 = A00(B01 - B11)
    #   M4 = A11(B10 - B00)
    #   M5 = (A00 + A01)B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    # Results:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    # For each sub-matrix multiplication at level 2, we further decompose into
    # schoolbook scalar multiplications.
    
    # Let's define the 7 Strassen combinations at level 1 for 6x6:
    # Split 6x6 into 2x2 blocks of 3x3:
    #   A = [[A00, A01],   B = [[B00, B01],   C = [[C00, C01],
    #         [A10, A11]]       [B10, B11]]       [C10, C11]]
    #
    # Strassen level 1:
    #   S1 = (A00+A11)(B00+B11)
    #   S2 = (A10+A11)B00
    #   S3 = A00(B01-B11)
    #   S4 = A11(B10-B00)
    #   S5 = (A00+A01)B11
    #   S6 = (A10-A00)(B00+B01)
    #   S7 = (A01-A11)(B10+B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Si is a 3x3 matrix product. We decompose each into 7 level-2 Strassen
    # terms, and each of those into schoolbook scalar multiplications.
    
    # For a 3x3 product decomposed via Strassen into 7 sub-products of sizes
    # (2,1)x(2,1), each sub-product has a specific shape. Let's enumerate.
    
    # Level 2 Strassen sub-products for computing C_block = A_block * B_block:
    # Each term involves products of sub-blocks. The sub-blocks have shapes:
    #   2x2 * 2x2 -> 4 scalar mults
    #   2x2 * 2x1 -> 4 scalar mults  
    #   2x1 * 2x1 -> 2 scalar mults
    #   2x1 * 1x1 -> 2 scalar mults
    #   1x2 * 2x2 -> 4 scalar mults
    #   1x2 * 2x1 -> 2 scalar mults
    #   1x1 * 2x2 -> 2 scalar mults
    #   1x1 * 2x1 -> 1 scalar mult
    #   1x1 * 1x1 -> 1 scalar mult
    #   2x2 * 1x1 -> 2 scalar mults
    #   1x2 * 1x2 -> 4 scalar mults
    #   1x2 * 1x1 -> 1 scalar mult
    #   2x1 * 1x2 -> 2 scalar mults (actually this is dot product, 1 mult... no)
    
    # This is getting complex. Let me just directly construct the rank-196
    # decomposition by building up from the known formula.
    
    # Alternative cleaner approach: directly build the 196 terms.
    # 
    # Total rank = 7 (level 1) * 7 (level 2) * 4 (schoolbook 2x2x2x2 base)
    # Wait, that's only if all sub-products are 2x2. But they aren't all.
    #
    # Actually, the standard analysis: 
    # Rank(M_{6,6,6}) <= 7 * Rank(M_{3,3,3})
    # Rank(M_{3,3,3}) <= 7 * 4 = 28? No, that's not right either.
    #
    # Let me think more carefully. The 3x3 Strassen split gives 7 multiplications
    # of matrices whose sizes vary. The total number of scalar multiplications
    # for one 3x3 product via this method:
    #   M1: (2x2+1x1) * (2x2+1x1) -> (3x3)*(3x3)? No, that's wrong.
    #
    # I'm overcomplicating. Let me just use the simple fact:
    # Rank(M_{6,6,6}) <= 7 * 7 * 4 = 196
    # by noting we can pad 3x3 to 4x4 (rank 28 for 4x4 via one Strassen level
    # on 2x2 blocks, where each 2x2 uses 4 scalar mults, so 7*4=28), and
    # 6x6 to 8x8? No.
    #
    # Simplest valid upper bound: 6x6 can be seen as embedding in 8x8, and
    # 8x8 has rank 7^3 = 343 via 3 levels of Strassen on 2x2. That's worse.
    #
    # Let me just do: pad 6 to 8, use 3-level Strassen on 8x8: rank 343. Worse.
    #
    # Or: pad 6 to 7? 7x7 via one Strassen level on 2x2 blocks of sizes (3,4):
    # rank 7 * max(Rank(3x3), Rank(4x4))... complex.
    #
    # Easiest safe improvement: just use schoolbook but that's rank 216 = seed.
    #
    # Let me try a different known result: R(6,6,6) <= 189 via recursive Strassen
    # on 6x6 = 2x2 blocks of 3x3, where each 3x3 uses rank 27 (schoolbook).
    # 7 * 27 = 189. This IS constructible!
    #
    # Level 1: 7 Strassen terms, each is a 3x3 matrix multiplication.
    # Each 3x3 multiplication is decomposed via schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
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
    # T1: (A00 + A11) * (B00 + B11)
    # T2: (A10 + A11) * B00
    # T3: A00 * (B01 - B11)
    # T4: A11 * (B10 - B00)
    # T5: (A00 + A01) * B11
    # T6: (A10 - A00) * (B00 + B01)
    # T7: (A01 - A11) * (B10 + B11)
    
    # Result blocks:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # For each Strassen term Tj, which computes X * Y where X and Y are 3x3
    # linear combinations of the original 3x3 blocks, we decompose into
    # schoolbook: for each (i,j,k) in 0..2, contribution to X_{i,a}*Y_{a,k}
    
    # Let me define for each Strassen term:
    # - The linear combination for X (left factor): which blocks with what coefficients
    # - The linear combination for Y (right factor): which blocks with what coefficients
    
    # Block indices: 00=(0:3,0:3), 01=(0:3,3:6), 10=(3:6,0:3), 11=(3:6,3:6)
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
        #
        # In the tensor decomposition, each such scalar mult contributes one
        # rank-1 term with:
        #   U entry: coefficient of A_{row,col} in x_{i,a}
        #   V entry: coefficient of B_{row,col} in y_{a,k}
        #   W entry: contributes to C_{block_row*3+i, block_col*3+k} with sign
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # x_{i,a} = sum over (br,bc) in left_combo: coeff * A_{3*br+i, 3*bc+a}
                    # y_{a,k} = sum over (br,bc) in right_combo: coeff * B_{3*br+a, 3*bc+k}
                    
                    # For U: entry at row (3*br+i)*6 + (3*bc+a) for each block in left_combo
                    # For V: entry at row (3*br+a)*6 + (3*bc+k) for each block in right_combo
                    # For W: this contributes to C at position determined by result blocks
                    
                    # U matrix: row index = i_A * m + j_A where i_A is row in A, j_A is col in A
                    # A has shape 6x6, so row index = i_A * 6 + j_A
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
                    # Wait, the convention is W[k*n+i, r] where output is C[i,k]?
                    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
                    # So W row index = k*n + i, meaning for output C[i,k], the row in W is k*n+i.
                    # And U row = i*m+j (input A[i,j]), V row = j*p+k (input B[j,k])
                    # The contraction is sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This should give C[i,k]
                    
                    # So for our term, it contributes to C[i_total, k_total] where
                    # i_total = 3*rbr + i, k_total = 3*rbc + k
                    # W row = k_total * n + i_total = (3*rbc+k)*6 + (3*rbr+i)
                    
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
