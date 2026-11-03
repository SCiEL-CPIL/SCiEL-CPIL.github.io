def construct():
    n, m, p = 4, 4, 6
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a scalar product
    # u_idx, v_idx, w_idx are the indices in U, V, W
    # val is the coefficient
    # col is the column index
    def add_product(col, u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Part 1: Left 2 columns (k=0, 1)
    # Rank 28
    # Strassen products s=0..6
    # Each product is 2x2 * 2x1 -> 2x1
    # Decomposed into 4 scalar products
    
    # Define Strassen coefficients for A (2x2 blocks)
    # alpha[s][bi][bj] is a 2x2 matrix of coefficients for A[bi*2+i, bj*2+j]
    # Actually, we can just define the linear forms.
    
    # s=0: L=A11+A22
    # s=1: L=A21+A22
    # s=2: L=A11
    # s=3: L=A22
    # s=4: L=A11+A12
    # s=5: L=A21-A11
    # s=6: L=A12-A22
    
    # s=0: R=B11+B22
    # s=1: R=B11
    # s=2: R=B12-B22
    # s=3: R=B21-B11
    # s=4: R=B22
    # s=5: R=B11+B12
    # s=6: R=B21+B22
    
    # For Part 1, B blocks are 2x1.
    # B11 is B[0:2, 0:2] -> No, B is 4x6.
    # Left part is B[:, 0:2].
    # B11 = B[0:2, 0:2] -> 2x2? No.
    # B is 4x6.
    # Blocks of B for Part 1:
    # B11 = B[0:2, 0:1] (2x1)
    # B12 = B[0:2, 1:2] (2x1)
    # B21 = B[2:4, 0:1] (2x1)
    # B22 = B[2:4, 1:2] (2x1)
    
    # Wait, the Strassen split for B (4x2) is into 2x2 blocks of size 2x1.
    # B11 (top-left), B12 (top-right), B21 (bottom-left), B22 (bottom-right).
    # Yes.
    
    # Coefficients for L_s (A side)
    # L_s is a 2x2 matrix.
    # L_s[i][j] = sum_{bi, bj} alpha_s[bi, bj, i, j] * A[bi*2+i, bj*2+j]
    # alpha_s is a dict of (bi, bj) -> 2x2 matrix
    
    alpha = {
        0: {(0,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]},
        1: {(1,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]},
        2: {(0,0): [[1,1],[1,1]]},
        3: {(1,1): [[1,1],[1,1]]},
        4: {(0,0): [[1,1],[1,1]], (0,1): [[1,1],[1,1]]},
        5: {(1,0): [[1,1],[1,1]], (0,0): [[-1,-1],[-1,-1]]},
        6: {(0,1): [[1,1],[1,1]], (1,1): [[-1,-1],[-1,-1]]}
    }
    
    # Coefficients for R_s (B side, 2x1)
    # R_s[i][k] = sum_{bj, bk} beta_s[bj, bk, i, k] * B[bj*2+i, bk*1+k]
    # Here bk is 0 or 1 (block row index in B).
    # k is 0 or 1 (column index within block).
    # So B index is (bj*2+i, bk*1+k) = (bj*2+i, bk+k).
    # Wait, B blocks are 2x1.
    # B11 is rows 0,1 cols 0.
    # B12 is rows 0,1 cols 1.
    # B21 is rows 2,3 cols 0.
    # B22 is rows 2,3 cols 1.
    
    beta = {
        0: {(0,0): [[1],[1]], (1,1): [[1],[1]]}, # B11+B22. Note: B11 is (0,0) block, B22 is (1,1) block.
        1: {(0,0): [[1],[1]]},
        2: {(0,1): [[1],[1]], (1,1): [[-1],[-1]]},
        3: {(1,0): [[1],[1]], (0,0): [[-1],[-1]]},
        4: {(1,1): [[1],[1]]},
        5: {(0,0): [[1],[1]], (0,1): [[1],[1]]},
        6: {(1,0): [[1],[1]], (1,1): [[1],[1]]}
    }
    
    # Gamma for Part 1 (C is 4x2, blocks 2x1)
    # C11 = M0 - M1 + M3
    # C12 = M2 + M4
    # C21 = M1 + M5
    # C22 = M0 - M2 + M4 + M6
    # gamma[s][bi][bk] is coefficient of M_s in C_block(bi, bk)
    # bi is 0,1 (row block of C), bk is 0,1 (col block of C)
    
    gamma = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    
    # Wait, the gamma indices in the previous code were:
    # s=0 (M1): C11(+1), C22(+1) -> gamma[0] = {(0,0): 1, (1,1): 1}
    # s=1 (M2): C21(+1), C22(-1) -> gamma[1] = {(1,0): 1, (1,1): -1}
    # s=2 (M3): C12(+1), C22(+1) -> gamma[2] = {(0,1): 1, (1,1): 1}
    # s=3 (M4): C11(+1), C21(+1) -> gamma[3] = {(0,0): 1, (1,0): 1}
    # s=4 (M5): C11(-1), C12(+1) -> gamma[4] = {(0,0): -1, (0,1): 1}
    # s=5 (M6): C22(+1) -> gamma[5] = {(1,1): 1}
    # s=6 (M7): C11(+1) -> gamma[6] = {(0,0): 1}
    
    # Let's re-verify the Strassen formulas.
    # C11 = M1 - M2 + M4
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 + M7
    
    # My gamma dict:
    # s=0 (M1): C11(+1), C22(+1). Correct.
    # s=1 (M2): C21(+1), C22(-1). Correct.
    # s=2 (M3): C12(+1), C22(-1)? No, C22 has -M3.
    # In my dict: gamma[2] = {(0,1): 1, (1,1): 1}. This says C22(+1).
    # But formula says C22 = M1 - M3 + M5 + M7. So C22 has -1 for M3.
    # So gamma[2] should be {(0,1): 1, (1,1): -1}.
    
    # Let's fix gamma.
    gamma = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): -1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    # Check s=4 (M5): C11(-1), C12(+1). Correct.
    # Check s=5 (M6): C22(+1). Correct.
    # Check s=6 (M7): C11(+1). Correct.
    
    # Now fill U, V, W for Part 1
    col_offset = 0
    for s in range(7):
        # For each of the 4 scalar products in the 2x2 * 2x1 multiplication
        for i in range(2): # row of result
            for j in range(2): # col of A (which is row of B)
                # This is not quite right.
                # The product M_s[i] = sum_{j=0}^1 L_s[i][j] * R_s[j][0]
                # So we have 2 outputs, each is sum of 2 products.
                # Total 4 scalar products.
                # Let's iterate over the 4 scalar products explicitly.
                pass
                
    # Better:
    # For each s, we have L_s (2x2) and R_s (2x1).
    # M_s[0] = L_s[0][0]*R_s[0][0] + L_s[0][1]*R_s[1][0]
    # M_s[1] = L_s[1][0]*R_s[0][0] + L_s[1][1]*R_s[1][0]
    
    # We will create 4 columns for each s.
    for s in range(7):
        for t in range(4):
            col = col_offset + s * 4 + t
            # Determine which term of the dot product this is
            # t=0: L[0][0]*R[0][0]
            # t=1: L[0][1]*R[1][0]
            # t=2: L[1][0]*R[0][0]
            # t=3: L[1][1]*R[1][0]
            
            # Get L_s and R_s coefficients
            # L_s[i][j] is sum of alpha_s[bi,bj][i][j] * A[bi*2+i, bj*2+j]
            # R_s[j][0] is sum of beta_s[bj,bk][j][0] * B[bj*2+j, bk]
            
            # For U:
            # We need to add to U[u_idx, col] the coefficient of A[u_idx] in L_s[i][j]
            # Where i, j are determined by t.
            
            if t == 0:
                i, j = 0, 0
            elif t == 1:
                i, j = 0, 1
            elif t == 2:
                i, j = 1, 0
            else:
                i, j = 1, 1
                
            # U part
            for (bi, bj), mat in alpha[s].items():
                val = mat[i][j]
                if val != 0:
                    u_idx = bi * 2 + i
                    # Wait, u_idx is index in A.
                    # A is 4x4.
                    # u_idx should be (bi*2+i) * 4 + (bj*2+j)?
                    # No, the term is A[bi*2+i, bj*2+j].
                    # So u_idx = (bi*2+i) * 4 + (bj*2+j).
                    u_idx = (bi * 2 + i) * 4 + (bj * 2 + j)
                    U[u_idx][col] += val
            
            # V part
            # R_s[j][0] involves B[bj*2+j, bk].
            # Wait, R_s is 2x1.
            # R_s[row][col] where col=0.
            # The term is R_s[j][0].
            # R_s[j][0] = sum_{bk} beta_s[bj,bk][j][0] * B[bj*2+j, bk]
            # Note: beta_s[bj,bk] is a 2x1 matrix.
            # So beta_s[bj,bk][j][0] is the coefficient.
            
            for (bj, bk), mat in beta[s].items():
                val = mat[j][0]
                if val != 0:
                    v_idx = (bj * 2 + j) * 6 + bk
                    V[v_idx][col] += val
            
            # W part
            # The result M_s[i] contributes to C blocks.
            # C_block(bi_c, bk_c) gets gamma[s][bi_c, bk_c] * M_s[i]
            # M_s[i] is the i-th row of the 2x1 result.
            # So i=0 corresponds to top row of the block, i=1 to bottom row.
            # The block is at (bi_c, bk_c).
            # The global row index in C is bi_c*2 + i.
            # The global col index in C is bk_c*1 + 0 = bk_c.
            # So w_idx = bk_c * 4 + (bi_c*2 + i).
            
            for (bi_c, bk_c), val in gamma[s].items():
                if val != 0:
                    w_idx = bk_c * 4 + (bi_c * 2 + i)
                    W[w_idx][col] += val
    
    col_offset += 28

    # Part 2: Right 4 columns (k=2, 3, 4, 5)
    # Rank 49
    # Strassen products s=0..6
    # Each product is 2x2 * 2x2 -> 2x2
    # Decomposed into 7 scalar products (Strassen)
    
    # B blocks for Part 2:
    # B11 = B[0:2, 2:4] (2x2)
    # B12 = B[0:2, 4:6] (2x2)
    # B21 = B[2:4, 2:4] (2x2)
    # B22 = B[2:4, 4:6] (2x2)
    
    # alpha is the same.
    # beta is different (2x2 matrices).
    
    beta2 = {
        0: {(0,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]},
        1: {(0,0): [[1,1],[1,1]]},
        2: {(0,1): [[1,1],[1,1]], (1,1): [[-1,-1],[-1,-1]]},
        3: {(1,0): [[1,1],[1,1]], (0,0): [[-1,-1],[-1,-1]]},
        4: {(1,1): [[1,1],[1,1]]},
        5: {(0,0): [[1,1],[1,1]], (0,1): [[1,1],[1,1]]},
        6: {(1,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]}
    }
    
    # gamma is the same (C blocks are 2x2).
    
    for s in range(7):
        # Strassen for 2x2 * 2x2
        # 7 scalar products
        # Let's define the 7 products for 2x2 Strassen.
        # Inputs: L (2x2), R (2x2)
        # L = [[l00, l01], [l10, l11]]
        # R = [[r00, r01], [r10, r11]]
        
        # P1 = (l00+l11)(r00+r11)
        # P2 = (l10+l11)r00
        # P3 = l00(r11-r01)
        # P4 = l11(r10-r00)
        # P5 = (l00+l01)r11
        # P6 = (l10-l00)(r00+r01)
        # P7 = (l01-l11)(r10+r11)
        
        # Results:
        # C00 = P1 - P2 + P4
        # C01 = P3 + P5
        # C10 = P2 + P6
        # C11 = P1 - P3 + P5 + P7
        
        # We need to decompose each P_t into scalar products of L and R entries.
        # Each P_t is a dot product of length 2.
        # So 7 * 2 = 14? No.
        # Wait, the rank of 2x2*2x2 is 7.
        # The 7 products P1..P7 are the scalar products.
        # Each P_t is a single scalar multiplication of two linear forms.
        # So we just need to identify the linear forms.
        
        for t in range(7):
            col = col_offset + s * 7 + t
            
            # Determine the linear forms for L and R for product t
            # We define a list of (i, j) pairs for L and (j, k) pairs for R?
            # No, L and R are 2x2 matrices.
            # The product is sum_{j} L[i][j] R[j][k].
            # But here L and R are already the combined blocks.
            # So L[i][j] is a linear form in A.
            # R[j][k] is a linear form in B.
            
            # We need to express P_t as a sum of products of linear forms in A and B.
            # Actually, P_t is just ONE product of two linear forms.
            # So we just need to find the coefficients of A and B in those forms.
            
            # Let's define the forms explicitly.
            # P1 = (l00+l11)(r00+r11)
            # Form L: l00 + l11
            # Form R: r00 + r11
            
            # We will define a helper to get the coefficients of a linear form.
            
            # For U:
            # We need to add to U the coefficients of A in the L form.
            # For V:
            # We need to add to V the coefficients of B in the R form.
            # For W:
            # We need to add to W the coefficient of P_t in the result C.
            
            # Let's define the L and R forms for each t.
            # L_form is a 2x2 matrix of coefficients for l_ij.
            # R_form is a 2x2 matrix of coefficients for r_jk.
            
            L_form = [[0]*2 for _ in range(2)]
            R_form = [[0]*2 for _ in range(2)]
            
            if t == 0:
                L_form[0][0] = 1; L_form[1][1] = 1
                R_form[0][0] = 1; R_form[1][1] = 1
            elif t == 1:
                L_form[1][0] = 1; L_form[1][1] = 1
                R_form[0][0] = 1
            elif t == 2:
                L_form[0][0] = 1
                R_form[1][1] = 1; R_form[0][1] = -1
            elif t == 3:
                L_form[1][1] = 1
                R_form[1][0] = 1; R_form[0][0] = -1
            elif t == 4:
                L_form[0][0] = 1; L_form[0][1] = 1
                R_form[1][1] = 1
            elif t == 5:
                L_form[1][0] = 1; L_form[0][0] = -1
                R_form[0][0] = 1; R_form[0][1] = 1
            else:
                L_form[0][1] = 1; L_form[1][1] = -1
                R_form[1][0] = 1; R_form[1][1] = 1
            
            # Now, L_form[i][j] is the coefficient of l_ij in the form.
            # l_ij is a linear form in A.
            # l_ij = sum_{bi,bj} alpha[s][bi,bj][i][j] A[bi*2+i, bj*2+j]
            
            # So the total coefficient of A[bi*2+i, bj*2+j] in the L form is:
            # sum_{i,j} L_form[i][j] * alpha[s][bi,bj][i][j]
            
            for bi in range(2):
                for bj in range(2):
                    if (bi, bj) in alpha[s]:
                        mat_a = alpha[s][(bi, bj)]
                        for i in range(2):
                            for j in range(2):
                                coeff = L_form[i][j] * mat_a[i][j]
                                if coeff != 0:
                                    u_idx = (bi * 2 + i) * 4 + (bj * 2 + j)
                                    U[u_idx][col] += coeff
            
            # Similarly for R.
            # r_jk = sum_{bj,bk} beta2[s][bj,bk][j][k] B[bj*2+j, bk*2+k]
            # Note: In Part 2, B blocks are 2x2.
            # So B index is (bj*2+j, bk*2+k).
            
            for bj in range(2):
                for bk in range(2):
                    if (bj, bk) in beta2[s]:
                        mat_b = beta2[s][(bj, bk)]
                        for j in range(2):
                            for k in range(2):
                                coeff = R_form[j][k] * mat_b[j][k]
                                if coeff != 0:
                                    v_idx = (bj * 2 + j) * 6 + (bk * 2 + k)
                                    V[v_idx][col] += coeff
            
            # Now W.
            # P_t contributes to C blocks.
            # C00 = P1 - P2 + P4
            # C01 = P3 + P5
            # C10 = P2 + P6
            # C11 = P1 - P3 + P5 + P7
            
            # gamma2[s] is the same as gamma[s] for the outer Strassen.
            # But wait, the inner Strassen is for the 2x2 block multiplication.
            # The result of the inner Strassen is a 2x2 matrix M_s.
            # M_s is then used in the outer Strassen combination.
            # So the contribution of P_t to M_s is determined by the inner Strassen gamma.
            
            # Inner Strassen Gamma:
            # g_in[t][i][j] is coefficient of P_t in M_s[i][j]
            
            g_in = {
                0: {(0,0): 1, (1,1): 1},
                1: {(1,0): 1, (1,1): -1},
                2: {(0,1): 1, (1,1): -1},
                3: {(0,0): 1, (1,0): 1},
                4: {(0,0): -1, (0,1): 1},
                5: {(1,1): 1},
                6: {(0,0): 1}
            }
            
            # M_s[i][j] is the (i,j) entry of the 2x2 block.
            # This block is at (bi_c, bk_c) in the outer C.
            # So M_s[i][j] corresponds to C[bi_c*2+i, bk_c*2+j].
            
            # For each (bi_c, bk_c) in gamma[s]:
            #   coeff = gamma[s][bi_c, bk_c]
            #   For each (i, j) in g_in[t]:
            #       w_coeff = coeff * g_in[t][i][j]
            #       w_idx = (bk_c*2+j) * 4 + (bi_c*2+i)
            
            for (bi_c, bk_c), coeff_outer in gamma[s].items():
                if coeff_outer != 0:
                    if t in g_in:
                        for (i, j), coeff_inner in g_in[t].items():
                            w_coeff = coeff_outer * coeff_inner
                            if w_coeff != 0:
                                w_idx = (bk_c * 2 + j) * 4 + (bi_c * 2 + i)
                                W[w_idx][col] += w_coeff

    return {"U": U, "V": V, "W": W}
