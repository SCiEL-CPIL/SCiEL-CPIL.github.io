"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Recursive Strassen: 4x4 = 7 block mults of 2x2, each 2x2 = 7 scalar mults
    # Total rank = 49
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # n*m = 16
    V = [[0] * rank for _ in range(16)]  # m*p = 16
    W = [[0] * rank for _ in range(16)]  # p*n = 16
    
    # Strassen 2x2 coefficients:
    # For multiplying X*Y where X,Y are 2x2:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y10-Y11)
    # S4 = X11(Y01-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # Block-level Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4, M1-M2+M3+M6]]
    
    # We need to expand each block product Mk = Xk * Yk into 7 scalar products.
    # Xk and Yk are 2x2 matrices whose entries are linear combinations of A and B entries.
    
    # Let's define the 7 block-level terms with their A and B coefficient vectors.
    # A is 4x4: A[i][j] where i,j in 0..3
    # B is 4x4: B[j][k] where j,k in 0..3
    # C is 4x4: C[i][k] where i,k in 0..3
    
    # Block indices: bi,bj for rows/cols of blocks, li,lj for local indices within blocks
    # Global index: i = 2*bi + li, j = 2*bj + lj
    
    # For each block term t (0..6), define:
    # A_terms[t]: list of (global_i, global_j, coeff) for the X matrix in X*Y
    # B_terms[t]: list of (global_j, global_k, coeff) for the Y matrix in X*Y
    
    # Actually, let me think differently. 
    # X is a 2x2 matrix where X[bi][bj] = sum of A[2*bi+li][2*bj+lj] * coeff
    # Similarly for Y.
    
    # Let me define the block-level Strassen terms:
    # Each term has X = linear combo of A blocks, Y = linear combo of B blocks.
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), 
    #            A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #            B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # Block-level definitions (which A blocks and B blocks to combine):
    # t=0 (M1): X = A00+A11, Y = B00+B11
    # t=1 (M2): X = A10+A11, Y = B00
    # t=2 (M3): X = A00, Y = B10-B11
    # t=3 (M4): X = A11, Y = B01-B00
    # t=4 (M5): X = A00+A01, Y = B11
    # t=5 (M6): X = A10-A00, Y = B00+B01
    # t=6 (M7): X = A01-A11, Y = B10+B11
    
    # For each block term t, X and Y are 2x2 matrices.
    # X[0][0] = sum of A entries from A00 with coeff 1 (for t=0,4) or 0
    # X[0][1] = sum of A entries from A01 with coeff 1 (for t=4) or 0
    # X[1][0] = sum of A entries from A10 with coeff 1 (for t=1,5) or 0
    # X[1][1] = sum of A entries from A11 with coeff 1 (for t=0,1,3,6) or 0
    
    # Let me be more explicit. For each t, define:
    # X_coeffs: 2x2 matrix where X_coeffs[bi][bj] = coefficient for block (bi,bj) of A
    # Y_coeffs: 2x2 matrix where Y_coeffs[bi][bj] = coefficient for block (bi,bj) of B
    
    X_coeffs = [
        [[1, 0], [0, 1]],   # t=0: X = A00 + A11
        [[0, 0], [1, 1]],   # t=1: X = A10 + A11
        [[1, 0], [0, 0]],   # t=2: X = A00
        [[0, 0], [0, 1]],   # t=3: X = A11
        [[1, 1], [0, 0]],   # t=4: X = A00 + A01
        [[0, 0], [1, -1]],  # t=5: X = A10 - A00
        [[0, 1], [0, -1]],  # t=6: X = A01 - A11
    ]
    
    Y_coeffs = [
        [[1, 0], [0, 1]],   # t=0: Y = B00 + B11
        [[1, 0], [0, 0]],   # t=1: Y = B00
        [[0, 0], [1, -1]],  # t=2: Y = B10 - B11
        [[0, 1], [0, -1]],  # t=3: Y = B01 - B00
        [[0, 0], [0, 1]],   # t=4: Y = B11
        [[1, 1], [0, 0]],   # t=5: Y = B00 + B01
        [[0, 0], [1, 1]],   # t=6: Y = B10 + B11
    ]
    
    # Now for each block term t, we apply Strassen to compute X*Y (both 2x2).
    # The 7 inner Strassen terms:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y10-Y11)
    # S4 = X11(Y01-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # For each inner term s (0..6), define the coefficients for X and Y entries.
    # X_entries: 2x2 matrix of coefficients for X[bi][bj]
    # Y_entries: 2x2 matrix of coefficients for Y[bi][bj]
    
    # S1: X00+X11 and Y00+Y11
    # S2: X10+X11 and Y00
    # S3: X00 and Y10-Y11
    # S4: X11 and Y01-Y00
    # S5: X00+X01 and Y11
    # S6: X10-X00 and Y00+Y01
    # S7: X01-X11 and Y10+Y11
    
    inner_X = [
        [[1, 0], [0, 1]],   # S1: X00+X11
        [[0, 0], [1, 1]],   # S2: X10+X11
        [[1, 0], [0, 0]],   # S3: X00
        [[0, 0], [0, 1]],   # S4: X11
        [[1, 1], [0, 0]],   # S5: X00+X01
        [[-1, 0], [1, 0]],  # S6: X10-X00
        [[0, 1], [0, -1]],  # S7: X01-X11
    ]
    
    inner_Y = [
        [[1, 0], [0, 1]],   # S1: Y00+Y11
        [[1, 0], [0, 0]],   # S2: Y00
        [[0, 0], [1, -1]],  # S3: Y10-Y11
        [[0, 1], [0, -1]],  # S4: Y01-Y00
        [[0, 0], [0, 1]],   # S5: Y11
        [[1, 1], [0, 0]],   # S6: Y00+Y01
        [[0, 0], [1, 1]],   # S7: Y10+Y11
    ]
    
    # C output coefficients:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each block term t, the contribution to C blocks:
    # C00: t=0:+1, t=3:+1, t=4:-1, t=6:+1
    # C01: t=2:+1, t=4:+1
    # C10: t=1:+1, t=3:+1
    # C11: t=0:+1, t=1:-1, t=2:+1, t=5:+1
    
    C_block_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        (0, 1): {2: 1, 4: 1},                 # C01
        (1, 0): {1: 1, 3: 1},                 # C10
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},   # C11
    }
    
    # Now build the 49 rank-1 terms.
    col = 0
    for t in range(7):  # block term
        for s in range(7):  # inner Strassen term
            # For this (t,s) pair, we have a rank-1 term.
            # U vector: coefficients for A entries
            # V vector: coefficients for B entries
            # W vector: coefficients for C entries
            
            # U[i][j] = sum over block positions (bi,bj) of X_coeffs[t][bi][bj] * inner_X[s][bi][bj]
            # Actually, X[bi][bj] is a 2x2 matrix of A entries.
            # X[bi][bj] = X_coeffs[t][bi][bj] * A_block(bi,bj)
            # And inner Strassen uses X entries:
            # X_entry[bi][bj] = inner_X[s][bi][bj] * X[bi][bj]
            # = inner_X[s][bi][bj] * X_coeffs[t][bi][bj] * A_block(bi,bj)
            
            # So the coefficient for A[i][j] where i=2*bi+li, j=2*bj+lj is:
            # inner_X[s][bi][bj] * X_coeffs[t][bi][bj]
            
            # Similarly for V:
            # V[j][k] where j=2*bi+li, k=2*bj+lj:
            # inner_Y[s][bi][bj] * Y_coeffs[t][bi][bj]
            
            # For W:
            # The product X*Y contributes to C blocks.
            # The inner Strassen result S_s contributes to the 2x2 matrix X*Y.
            # Strassen output:
            # (X*Y)00 = S1 + S4 - S5 + S7
            # (X*Y)01 = S3 + S5
            # (X*Y)10 = S2 + S4
            # (X*Y)11 = S1 - S2 + S3 + S6
            
            # So S_s contributes to (X*Y)[ci][cj] with coefficient:
            # s=0 (S1): (0,0):+1, (1,1):+1
            # s=1 (S2): (1,0):+1, (1,1):-1
            # s=2 (S3): (0,1):+1, (1,1):+1
            # s=3 (S4): (0,0):+1, (1,0):+1
            # s=4 (S5): (0,0):-1, (0,1):+1
            # s=5 (S6): (1,1):+1
            # s=6 (S7): (0,0):+1
            
            S_to_block = {
                0: {(0,0): 1, (1,1): 1},     # S1
                1: {(1,0): 1, (1,1): -1},    # S2
                2: {(0,1): 1, (1,1): 1},     # S3
                3: {(0,0): 1, (1,0): 1},     # S4
                4: {(0,0): -1, (0,1): 1},    # S5
                5: {(1,1): 1},               # S6
                6: {(0,0): 1},               # S7
            }
            
            # Now for each (ci,cj) in S_to_block[s], the contribution to C block (ci,cj)
            # is S_to_block[s][(ci,cj)] * (the S_s value)
            # And C block (ci,cj) gets contribution from block term t with coefficient C_block_coeffs[(ci,cj)][t]
            
            # So W coefficient for C[2*ci+li][2*cj+lj] is:
            # C_block_coeffs[(ci,cj)][t] * S_to_block[s][(ci,cj)]
            
            # Build U:
            for bi in range(2):
                for bj in range(2):
                    coeff = inner_X[s][bi][bj] * X_coeffs[t][bi][bj]
                    if coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                i = 2*bi + li
                                j = 2*bj + lj
                                U[i*m + j][col] += coeff
            
            # Build V:
            for bi in range(2):
                for bj in range(2):
                    coeff = inner_Y[s][bi][bj] * Y_coeffs[t][bi][bj]
                    if coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                j = 2*bi + li
                                k = 2*bj + lj
                                V[j*p + k][col] += coeff
            
            # Build W:
            for (ci, cj), s_coeff in S_to_block[s].items():
                if t in C_block_coeffs[(ci, cj)]:
                    c_coeff = C_block_coeffs[(ci, cj)][t]
                    total_coeff = c_coeff * s_coeff
                    if total_coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                i = 2*ci + li
                                k = 2*cj + lj
                                W[k*n + i][col] += total_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
