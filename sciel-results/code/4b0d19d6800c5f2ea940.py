def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 Strassen decomposition (Rank 49)
    # We will use a recursive helper to generate the factors.
    # For now, we can hardcode the 2x2 Strassen and nest it.
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen 2x2x2 decomposition
    # Indices for 2x2:
    # A: 0,1,2,3 (00,01,10,11)
    # B: 0,1,2,3 (00,01,10,11)
    # C: 0,1,2,3 (00,01,10,11)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # We need to map these to the 4x4 case.
    # The 4x4 matrix is divided into 2x2 blocks.
    # A00 is 2x2, A01 is 2x2, etc.
    # Each 2x2 block multiplication requires 7 scalar multiplications.
    # Total 7 * 7 = 49.
    
    # We will iterate over the 7 Strassen steps for the 4x4 level.
    # For each step, we need to compute products of 2x2 matrices.
    # Each 2x2 product is computed using 7 scalar multiplications.
    
    # This is complex to generate dynamically without a library.
    # I will use a known explicit 49-rank decomposition if available, or generate it.
    
    # Given the constraints, I will use the 65 rank solution with a simpler 4x4x4 decomposition if possible,
    # or just use the 16+16+16+16 = 64 schoolbook for 4x4x4? No, 49 is better.
    
    # Let's use the fact that we can just use the schoolbook for the 4x4x4 part if we can't easily generate Strassen.
    # Schoolbook 4x4x4 is rank 64.
    # 64 + 16 = 80. This is the seed.
    
    # I need a rank < 80.
    # 49 + 16 = 65 is the target.
    
    # I will implement the 2x2 Strassen and nest it.
    
    # Define the 2x2 Strassen factors
    # U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Col 0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = -1; W2[2][0] = -1; W2[3][0] = 1
    
    # Col 1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 0 # C10 = M2 + M4. C11 = ...
    # Wait, W2 maps to C.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # M2 contributes to C10 (1) and C11 (0)?
    # C11 = M1 - M3 + M6 - M7. M2 is not in C11.
    # So W2[2][1] = 1, W2[3][1] = 0.
    
    # Col 2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # Col 3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = -1; W2[2][3] = 1
    
    # Col 4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[3][4] = 0 # M5 in C01(1), C11(0)
    
    # Col 5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = -1; W2[3][5] = 1
    
    # Col 6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Now nest for 4x4.
    # We have 7 "super-terms".
    # Each super-term is a 2x2 matrix multiplication.
    # So we need 7 * 7 = 49 columns.
    
    col_offset = 0
    for s in range(7):
        # Determine the U, V factors for the s-th super-term
        # U_super[s] is a 2x2 matrix of coefficients for A blocks
        # V_super[s] is a 2x2 matrix of coefficients for B blocks
        
        # M1: (A00+A11)(B00+B11) -> U=[[1,0],[0,1]], V=[[1,0],[0,1]]
        # M2: (A10+A11)B00 -> U=[[0,0],[1,1]], V=[[1,0],[0,0]]
        # M3: A00(B01-B11) -> U=[[1,0],[0,0]], V=[[0,1],[0,-1]]
        # M4: A11(B10-B00) -> U=[[0,0],[0,1]], V=[[-1,0],[1,0]]
        # M5: (A00+A01)B11 -> U=[[1,1],[0,0]], V=[[0,0],[0,1]]
        # M6: (A10-A00)(B00+B01) -> U=[[-1,0],[1,0]], V=[[1,1],[0,0]]
        # M7: (A01-A11)(B10+B11) -> U=[[0,1],[-1,0]], V=[[0,0],[1,1]]
        
        U_super = [
            [[1,0],[0,1]],
            [[0,0],[1,1]],
            [[1,0],[0,0]],
            [[0,0],[0,1]],
            [[1,1],[0,0]],
            [[-1,0],[1,0]],
            [[0,1],[-1,0]]
        ]
        V_super = [
            [[1,0],[0,1]],
            [[1,0],[0,0]],
            [[0,1],[0,-1]],
            [[-1,0],[1,0]],
            [[0,0],[0,1]],
            [[1,1],[0,0]],
            [[0,0],[1,1]]
        ]
        
        for t in range(7):
            # We are computing the t-th scalar product for the s-th super-term
            # U2[t] gives coefficients for the 2x2 A block
            # V2[t] gives coefficients for the 2x2 B block
            # W2[t] gives coefficients for the 2x2 C block
            
            # Map to 4x4 indices
            # A block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1
            # Index in U: (br*2 + r) * 2 + (bc*2 + c) ? No.
            # U is flattened row-major: i*m + j.
            # For A (4x4), index is i*4 + j.
            
            for r in range(2):
                for c in range(2):
                    # A block element (r,c) in block (br,bc)
                    # We need to sum over the 2x2 blocks
                    # U_super[s][br][bc] is the weight for block (br,bc)
                    # U2[t][r*2+c] is the weight for element (r,c) within the block
                    
                    # Wait, U2[t] is for a 2x2 matrix.
                    # The "A" in the 2x2 multiplication is the super-matrix.
                    # So U2[t][idx] corresponds to A[idx].
                    # A[idx] is a 2x2 block.
                    # So we multiply U2[t][idx] by U_super[s][br][bc] and add to U[global_idx].
                    
                    # Actually, U2[t] has 4 entries.
                    # Entry 0: A00, 1: A01, 2: A10, 3: A11.
                    # These are the 2x2 blocks.
                    
                    pass # This loop is getting complicated.
                    
            # Let's do it more directly.
            # For each of the 4 block positions (0,1,2,3) in the 2x2 super-matrix:
            for block_idx in range(4):
                # block_idx 0: A00, 1: A01, 2: A10, 3: A11
                # U2[t][block_idx] is the coefficient
                coeff_u = U2[t][block_idx]
                if coeff_u == 0: continue
                
                # Determine which 2x2 block of A this is
                br = block_idx // 2
                bc = block_idx % 2
                
                # U_super[s][br][bc] is the weight for this block in the s-th super-term
                weight = U_super[s][br][bc]
                final_coeff_u = coeff_u * weight
                if final_coeff_u == 0: continue
                
                # Add to U
                # The block (br, bc) in 4x4 A consists of 4 elements.
                # We need to add final_coeff_u * U2[t][sub_idx] to each?
                # No, U2[t] is already the coefficient for the block.
                # Wait, U2[t] is the factor for the 2x2 multiplication.
                # The 2x2 multiplication inputs are 2x2 matrices.
                # So U2[t][0] is the coefficient for the top-left 2x2 block.
                # But U2[t] is a vector of length 4.
                # This means the 2x2 multiplication is treated as a 4x4 scalar multiplication?
                # No.
                
                # Correct approach:
                # The 4x4 multiplication is composed of 7 2x2 multiplications.
                # Each 2x2 multiplication has rank 7.
                # So we have 49 scalar multiplications.
                # For the k-th scalar multiplication (k=0..48):
                # s = k // 7 (super-term)
                # t = k % 7 (sub-term)
                
                # U[k] is the coefficient for A elements.
                # V[k] is the coefficient for B elements.
                # W[k] is the coefficient for C elements.
                
                # U[k] is formed by:
                # For each block (br, bc) in A:
                #   weight = U_super[s][br][bc] * U2[t][block_idx]
                #   If weight != 0:
                #     For each element (r, c) in the block:
                #       U[br*2+r, bc*2+c] += weight * (1) ?
                #       No, U2[t][block_idx] is the coefficient for the whole block?
                #       No, U2 is for 2x2 scalar multiplication.
                #       So U2[t][0] is the coefficient for A00 (scalar).
                #       But here A00 is a 2x2 matrix.
                #       So we need to apply U2 to the elements of the block.
                
                # This implies we need to "expand" the 2x2 factors into 4x4 factors.
                # U2[t] has 4 entries.
                # Entry 0 corresponds to A00 (top-left of 2x2).
                # In the 4x4 context, A00 is a 2x2 block.
                # So U2[t][0] applies to all 4 elements of the block?
                # No, that would be wrong.
                # The 2x2 multiplication $M_{2x2}$ takes 2x2 matrices.
                # The factors U2, V2, W2 are for the scalar entries of those 2x2 matrices.
                # So if we have a 2x2 matrix $A_{block}$, the factor for its (0,0) entry is U2[t][0].
                # The factor for its (0,1) entry is U2[t][1].
                # Etc.
                
                # So:
                # For block (br, bc):
                #   For r in 0..1, c in 0..1:
                #     sub_idx = r*2 + c
                #     coeff = U_super[s][br][bc] * U2[t][sub_idx]
                #     U[br*2+r, bc*2+c] += coeff
                
                for r in range(2):
                    for c in range(2):
                        sub_idx = r * 2 + c
                        coeff = weight * U2[t][sub_idx]
                        if coeff != 0:
                            global_i = br * 2 + r
                            global_j = bc * 2 + c
                            U[global_i * 4 + global_j][col_offset + k] += coeff
                            
            # Similarly for V
            for block_idx in range(4):
                coeff_v = V2[t][block_idx]
                if coeff_v == 0: continue
                br = block_idx // 2
                bc = block_idx % 2
                weight = V_super[s][br][bc]
                final_coeff_v = coeff_v * weight
                if final_coeff_v == 0: continue
                for r in range(2):
                    for c in range(2):
                        sub_idx = r * 2 + c
                        coeff = weight * V2[t][sub_idx]
                        if coeff != 0:
                            global_j = br * 2 + r
                            global_k = bc * 2 + c
                            V[global_j * 4 + global_k][col_offset + k] += coeff
                            
            # Similarly for W
            # W maps to C blocks.
            for block_idx in range(4):
                coeff_w = W2[t][block_idx]
                if coeff_w == 0: continue
                br = block_idx // 2
                bc = block_idx % 2
                weight = W2[t][block_idx] # W2 is the same for all s?
                # No, W2 is the reconstruction for the 2x2 product.
                # The s-th super-term contributes to the C blocks based on W_super?
                # No, the s-th super-term IS the 2x2 product.
                # So the result of the s-th super-term is a 2x2 matrix of values.
                # These values are added to the corresponding C blocks.
                # So W_super is not needed. The W2 factors directly map to C blocks.
                # But we need to map the 2x2 C block to the 4x4 C.
                
                # Wait, W2[t][block_idx] is the coefficient for C_block[block_idx].
                # So:
                for r in range(2):
                    for c in range(2):
                        sub_idx = r * 2 + c
                        coeff = W2[t][sub_idx]
                        if coeff != 0:
                            global_k = br * 2 + r
                            global_i = bc * 2 + c
                            W[global_k * 4 + global_i][col_offset + k] += coeff

            k += 1

    # Part 2: 5th column
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49 + i * 4 + j] = 1
            V[j * 5 + 4][49 + i * 4 + j] = 1
            W[4 * 4 + i][49 + i * 4 + j] = 1

    return {"U": U, "V": V, "W": W}
