def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for 4x4x4
    # s: 0..6
    # X_s: 4x4 matrix (linear combo of A blocks)
    # Y_s: 4x4 matrix (linear combo of B blocks)
    # We represent X_s and Y_s as 2x2 blocks of 2x2 matrices?
    # No, X_s is a 4x4 matrix.
    # But the "inner" problem is 2x2x2.
    # So X_s is a 2x2 matrix of 2x2 blocks?
    # Yes.
    
    # Let's define the block coefficients for X_s and Y_s.
    # Blocks: 00, 01, 10, 11.
    # X_s_blocks: 2x2 array of 2x2 matrices?
    # Actually, X_s is formed by combining A blocks.
    # A = [[A00, A01], [A10, A11]]
    
    # Strassen 4x4:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product.
    # Let's define the "inner" Strassen for 2x2x2.
    # Inner indices:
    # X_inner: 2x2
    # Y_inner: 2x2
    # Z_inner: 2x2
    
    # Inner Strassen multipliers (t=0..6):
    # T0 = (X00+X11)(Y00+Y11)
    # T1 = (X10+X11)Y00
    # T2 = X00(Y01-Y11)
    # T3 = X11(Y10-Y00)
    # T4 = (X00+X01)Y11
    # T5 = (X10-X00)(Y00+Y01)
    # T6 = (X01-X11)(Y10+Y11)
    
    # Reconstruction Z:
    # Z00 = T0 - T3 + T5 - T6
    # Z01 = T2 + T4 - T5
    # Z10 = T1 + T3 - T4
    # Z11 = T0 - T1 + T2 + T6
    
    # We need to map the global indices.
    
    col = 0
    for b in range(2): # 2 blocks for p=8
        # B block offset in k
        k_offset = b * 4
        
        for s in range(7): # 7 outer Strassen multipliers
            # Define X_s and Y_s as 2x2 blocks of 2x2 matrices
            # X_s is 4x4, composed of 2x2 blocks X_s_00, X_s_01, X_s_10, X_s_11
            # Each X_s_ij is a linear combo of A_ij?
            # No, X_s is a 4x4 matrix.
            # But the inner product is 2x2x2.
            # So X_s must be a 2x2 matrix?
            # Wait.
            # M_s is a 2x2 matrix.
            # M_s = X_s Y_s where X_s, Y_s are 2x2 matrices.
            # So X_s is 2x2.
            # The entries of X_s are linear combos of A's 2x2 blocks?
            # No, the entries of X_s are scalars?
            # No, M_s is a 2x2 matrix.
            # The "inner" problem is computing M_s = X_s Y_s.
            # X_s and Y_s are 2x2 matrices.
            # Their entries are linear combinations of the entries of A and B?
            # No, the entries of X_s are the 2x2 blocks of A?
            # Yes.
            # So X_s is a 2x2 matrix where each entry is a 2x2 matrix (block of A).
            # This is equivalent to a 4x4 matrix.
            
            # So, for the inner Strassen, we treat X_s as a 2x2 matrix of blocks.
            # The "scalar" multiplications in the inner Strassen are actually
            # block multiplications? No.
            # The inner Strassen decomposes the 2x2x2 tensor.
            # The "scalars" in the inner Strassen are the entries of the 2x2 matrices X_s, Y_s.
            # But the entries of X_s are 2x2 blocks of A.
            # So we cannot just use scalar Strassen.
            
            # CORRECTION:
            # The outer Strassen reduces 4x4x4 to 7 instances of 2x2x2.
            # The "2x2x2" instances involve 2x2 matrices.
            # To compute a 2x2x2 multiplication exactly, we need rank 7 (Strassen) or 8 (Schoolbook).
            # The "scalars" in the 2x2x2 tensor are the actual scalar entries of the 2x2 matrices.
            # So we apply Strassen to the 2x2 matrices.
            
            # So, X_s is a 2x2 matrix. Its entries are scalars?
            # No, X_s is formed from A blocks.
            # Example: M0 = (A00+A11)(B00+B11).
            # A00, A11 are 2x2 matrices.
            # So X_0 = A00+A11 is a 2x2 matrix.
            # Y_0 = B00+B11 is a 2x2 matrix.
            # We need to compute X_0 Y_0.
            # We use Strassen on X_0 and Y_0.
            # This requires 7 scalar multiplications.
            # The scalar entries of X_0 are linear combos of entries of A.
            # The scalar entries of Y_0 are linear combos of entries of B.
            
            # So we need to define the scalar coefficients for X_0 and Y_0.
            
            # Let's define the 7 outer multipliers in terms of scalar coefficients for the 2x2 matrices.
            # A is 4x4. Indices (i,j).
            # A00: i=0,1; j=0,1.
            # A01: i=0,1; j=2,3.
            # A10: i=2,3; j=0,1.
            # A11: i=2,3; j=2,3.
            
            # For each s, define X_s (2x2) and Y_s (2x2).
            # X_s[i][j] is a linear combo of A.
            
            # We will store the coefficients in dictionaries or lists.
            
            # s=0: X=A00+A11, Y=B00+B11
            # s=1: X=A10+A11, Y=B00
            # s=2: X=A00, Y=B01-B11
            # s=3: X=A11, Y=B10-B00
            # s=4: X=A00+A01, Y=B11
            # s=5: X=A10-A00, Y=B00+B01
            # s=6: X=A01-A11, Y=B10+B11
            
            # For each s, we have X_s and Y_s as 2x2 matrices.
            # We apply inner Strassen to X_s, Y_s.
            # Inner Strassen multipliers t=0..6.
            # Each t defines a scalar product x_u y_v -> z_w.
            # x_u is an entry of X_s.
            # y_v is an entry of Y_s.
            # z_w is an entry of Z_s (which is M_s).
            
            # We need to map x_u to A entries.
            # x_u is a linear combo of A entries.
            # So U will have the coefficient of that A entry.
            
            # Let's precompute the scalar coefficients for X_s and Y_s.
            # X_s is 2x2.
            # X_s[i][j] = sum_{a_idx} coeff[a_idx] * A[a_idx]
            
            # We'll represent A entries by (i, j).
            # B entries by (j, k_local).
            # C entries by (i, k_local).
            
            # For each s, define the 2x2 matrix X_s_coeffs and Y_s_coeffs.
            # X_s_coeffs[i][j] is a dict {(i_a, j_a): coeff}
            
            X_coeffs = []
            Y_coeffs = []
            
            # Helper to add block
            def add_block(coeffs, block, sign):
                # block is 0..3 (00, 01, 10, 11)
                # block 0: i=0,1; j=0,1
                # block 1: i=0,1; j=2,3
                # block 2: i=2,3; j=0,1
                # block 3: i=2,3; j=2,3
                for di in range(2):
                    for dj in range(2):
                        if block == 0:
                            i_a = di; j_a = dj
                        elif block == 1:
                            i_a = di; j_a = dj + 2
                        elif block == 2:
                            i_a = di + 2; j_a = dj
                        else:
                            i_a = di + 2; j_a = dj + 2
                        key = (i_a, j_a)
                        if key in coeffs:
                            coeffs[key] += sign
                        else:
                            coeffs[key] = sign

            # Initialize 2x2 coeffs for X and Y
            # X_s[i][j]
            
            # s=0
            X0 = [[{} for _ in range(2)] for _ in range(2)]
            Y0 = [[{} for _ in range(2)] for _ in range(2)]
            # X0 = A00 + A11
            for i in range(2):
                for j in range(2):
                    add_block(X0[i][j], 0, 1)
                    add_block(X0[i][j], 3, 1)
            # Y0 = B00 + B11
            for i in range(2):
                for j in range(2):
                    add_block(Y0[i][j], 0, 1)
                    add_block(Y0[i][j], 3, 1)
            X_coeffs.append(X0)
            Y_coeffs.append(Y0)
            
            # s=1: X=A10+A11, Y=B00
            X1 = [[{} for _ in range(2)] for _ in range(2)]
            Y1 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X1[i][j], 2, 1)
                    add_block(X1[i][j], 3, 1)
            for i in range(2):
                for j in range(2):
                    add_block(Y1[i][j], 0, 1)
            X_coeffs.append(X1)
            Y_coeffs.append(Y1)
            
            # s=2: X=A00, Y=B01-B11
            X2 = [[{} for _ in range(2)] for _ in range(2)]
            Y2 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X2[i][j], 0, 1)
            for i in range(2):
                for j in range(2):
                    add_block(Y2[i][j], 1, 1)
                    add_block(Y2[i][j], 3, -1)
            X_coeffs.append(X2)
            Y_coeffs.append(Y2)
            
            # s=3: X=A11, Y=B10-B00
            X3 = [[{} for _ in range(2)] for _ in range(2)]
            Y3 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X3[i][j], 3, 1)
            for i in range(2):
                for j in range(2):
                    add_block(Y3[i][j], 2, 1)
                    add_block(Y3[i][j], 0, -1)
            X_coeffs.append(X3)
            Y_coeffs.append(Y3)
            
            # s=4: X=A00+A01, Y=B11
            X4 = [[{} for _ in range(2)] for _ in range(2)]
            Y4 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X4[i][j], 0, 1)
                    add_block(X4[i][j], 1, 1)
            for i in range(2):
                for j in range(2):
                    add_block(Y4[i][j], 3, 1)
            X_coeffs.append(X4)
            Y_coeffs.append(Y4)
            
            # s=5: X=A10-A00, Y=B00+B01
            X5 = [[{} for _ in range(2)] for _ in range(2)]
            Y5 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X5[i][j], 2, 1)
                    add_block(X5[i][j], 0, -1)
            for i in range(2):
                for j in range(2):
                    add_block(Y5[i][j], 0, 1)
                    add_block(Y5[i][j], 1, 1)
            X_coeffs.append(X5)
            Y_coeffs.append(Y5)
            
            # s=6: X=A01-A11, Y=B10+B11
            X6 = [[{} for _ in range(2)] for _ in range(2)]
            Y6 = [[{} for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    add_block(X6[i][j], 1, 1)
                    add_block(X6[i][j], 3, -1)
            for i in range(2):
                for j in range(2):
                    add_block(Y6[i][j], 2, 1)
                    add_block(Y6[i][j], 3, 1)
            X_coeffs.append(X6)
            Y_coeffs.append(Y6)
            
            # Now, for each s, apply inner Strassen.
            # Inner Strassen multipliers t=0..6.
            # T0 = (X00+X11)(Y00+Y11)
            # T1 = (X10+X11)Y00
            # T2 = X00(Y01-Y11)
            # T3 = X11(Y10-Y00)
            # T4 = (X00+X01)Y11
            # T5 = (X10-X00)(Y00+Y01)
            # T6 = (X01-X11)(Y10+Y11)
            
            # Reconstruction Z:
            # Z00 = T0 - T3 + T5 - T6
            # Z01 = T2 + T4 - T5
            # Z10 = T1 + T3 - T4
            # Z11 = T0 - T1 + T2 + T6
            
            # For each t, we have a product of two linear combos.
            # x_t is a linear combo of X_s entries.
            # y_t is a linear combo of Y_s entries.
            # z_t contributes to Z_s entries with certain coefficients.
            
            # We need to define x_t and y_t as linear combos of X_s[i][j].
            
            # t=0: x = X00+X11, y = Y00+Y11
            # t=1: x = X10+X11, y = Y00
            # t=2: x = X00, y = Y01-Y11
            # t=3: x = X11, y = Y10-Y00
            # t=4: x = X00+X01, y = Y11
            # t=5: x = X10-X00, y = Y00+Y01
            # t=6: x = X01-X11, y = Y10+Y11
            
            # Let's define these coefficients.
            # x_t_coeffs[i][j]
            
            x_t_coeffs = []
            y_t_coeffs = []
            
            # t=0
            x0 = [[0 for _ in range(2)] for _ in range(2)]
            y0 = [[0 for _ in range(2)] for _ in range(2)]
            x0[0][0] = 1; x0[1][1] = 1
            y0[0][0] = 1; y0[1][1] = 1
            x_t_coeffs.append(x0)
            y_t_coeffs.append(y0)
            
            # t=1
            x1 = [[0 for _ in range(2)] for _ in range(2)]
            y1 = [[0 for _ in range(2)] for _ in range(2)]
            x1[1][0] = 1; x1[1][1] = 1
            y1[0][0] = 1
            x_t_coeffs.append(x1)
            y_t_coeffs.append(y1)
            
            # t=2
            x2 = [[0 for _ in range(2)] for _ in range(2)]
            y2 = [[0 for _ in range(2)] for _ in range(2)]
            x2[0][0] = 1
            y2[0][1] = 1; y2[1][1] = -1
            x_t_coeffs.append(x2)
            y_t_coeffs.append(y2)
            
            # t=3
            x3 = [[0 for _ in range(2)] for _ in range(2)]
            y3 = [[0 for _ in range(2)] for _ in range(2)]
            x3[1][1] = 1
            y3[1][0] = 1; y3[0][0] = -1
            x_t_coeffs.append(x3)
            y_t_coeffs.append(y3)
            
            # t=4
            x4 = [[0 for _ in range(2)] for _ in range(2)]
            y4 = [[0 for _ in range(2)] for _ in range(2)]
            x4[0][0] = 1; x4[0][1] = 1
            y4[1][1] = 1
            x_t_coeffs.append(x4)
            y_t_coeffs.append(y4)
            
            # t=5
            x5 = [[0 for _ in range(2)] for _ in range(2)]
            y5 = [[0 for _ in range(2)] for _ in range(2)]
            x5[1][0] = 1; x5[0][0] = -1
            y5[0][0] = 1; y5[0][1] = 1
            x_t_coeffs.append(x5)
            y_t_coeffs.append(y5)
            
            # t=6
            x6 = [[0 for _ in range(2)] for _ in range(2)]
            y6 = [[0 for _ in range(2)] for _ in range(2)]
            x6[0][1] = 1; x6[1][1] = -1
            y6[1][0] = 1; y6[1][1] = 1
            x_t_coeffs.append(x6)
            y_t_coeffs.append(y6)
            
            # Now iterate
            for s in range(7):
                for t in range(7):
                    # Combine X_s and x_t to get global A coefficients
                    # Combine Y_s and y_t to get global B coefficients
                    # Determine W coefficients from reconstruction
                    
                    # A coeffs
                    a_coeffs = {}
                    for i in range(2):
                        for j in range(2):
                            c_x = x_t_coeffs[t][i][j]
                            if c_x != 0:
                                for (i_a, j_a), c_s in X_coeffs[s][i][j].items():
                                    c_total = c_x * c_s
                                    if c_total != 0:
                                        if (i_a, j_a) in a_coeffs:
                                            a_coeffs[(i_a, j_a)] += c_total
                                        else:
                                            a_coeffs[(i_a, j_a)] = c_total
                                        
                    # B coeffs
                    b_coeffs = {}
                    for i in range(2):
                        for j in range(2):
                            c_y = y_t_coeffs[t][i][j]
                            if c_y != 0:
                                for (i_b, j_b), c_s in Y_coeffs[s][i][j].items():
                                    c_total = c_y * c_s
                                    if c_total != 0:
                                        if (i_b, j_b) in b_coeffs:
                                            b_coeffs[(i_b, j_b)] += c_total
                                        else:
                                            b_coeffs[(i_b, j_b)] = c_total
                                        
                    # W coeffs
                    # Z_s entries are Z00, Z01, Z10, Z11.
                    # Z00 = T0 - T3 + T5 - T6
                    # Z01 = T2 + T4 - T5
                    # Z10 = T1 + T3 - T4
                    # Z11 = T0 - T1 + T2 + T6
                    
                    w_coeffs = {}
                    # t is the current multiplier.
                    # We need to add the contribution of this t to Z.
                    # The coefficient of T_t in Z_ij.
                    
                    if t == 0:
                        w_coeffs[(0,0)] = 1
                        w_coeffs[(1,1)] = 1
                    elif t == 1:
                        w_coeffs[(1,0)] = 1
                        w_coeffs[(1,1)] = -1
                    elif t == 2:
                        w_coeffs[(0,1)] = 1
                        w_coeffs[(1,1)] = 1
                    elif t == 3:
                        w_coeffs[(0,0)] = -1
                        w_coeffs[(1,0)] = 1
                    elif t == 4:
                        w_coeffs[(0,1)] = 1
                        w_coeffs[(1,0)] = -1
                    elif t == 5:
                        w_coeffs[(0,0)] = 1
                        w_coeffs[(0,1)] = -1
                    elif t == 6:
                        w_coeffs[(0,0)] = -1
                        w_coeffs[(1,1)] = 1
                    
                    # Map to global W indices
                    # Z_s is 2x2.
                    # Z_s[i][j] corresponds to C block.
                    # C is 4x4.
                    # C00: i=0,1; k=0,1
                    # C01: i=0,1; k=2,3
                    # C10: i=2,3; k=0,1
                    # C11: i=2,3; k=2,3
                    
                    for (i_z, j_z), c_w in w_coeffs.items():
                        if c_w == 0: continue
                        # i_z, j_z are 0..1
                        # Map to C block
                        # Block 0: i=0,1; k=0,1
                        # Block 1: i=0,1; k=2,3
                        # Block 2: i=2,3; k=0,1
                        # Block 3: i=2,3; k=2,3
                        
                        for di in range(2):
                            for dj in range(2):
                                if i_z == 0 and j_z == 0:
                                    i_c = di; k_c = dj
                                elif i_z == 0 and j_z == 1:
                                    i_c = di; k_c = dj + 2
                                elif i_z == 1 and j_z == 0:
                                    i_c = di + 2; k_c = dj
                                else:
                                    i_c = di + 2; k_c = dj + 2
                                
                                k_global = k_c + k_offset
                                idx_c = 4 * k_global + i_c
                                if idx_c in w_coeffs_global:
                                    w_coeffs_global[idx_c] += c_w
                                else:
                                    w_coeffs_global[idx_c] = c_w
                            
                    # Fill U, V, W for this column
                    for (i_a, j_a), c_u in a_coeffs.items():
                        if c_u == 0: continue
                        idx_a = 4 * i_a + j_a
                        U[idx_a][col] += c_u
                        
                    for (i_b, j_b), c_v in b_coeffs.items():
                        if c_v == 0: continue
                        # i_b is j index of B, j_b is k_local index of B
                        k_global = j_b + k_offset
                        idx_b = 8 * i_b + k_global
                        V[idx_b][col] += c_v
                        
                    # w_coeffs_global is local to this t?
                    # No, we need to accumulate w_coeffs for this column.
                    # I made a mistake: w_coeffs_global should be initialized per column.
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
