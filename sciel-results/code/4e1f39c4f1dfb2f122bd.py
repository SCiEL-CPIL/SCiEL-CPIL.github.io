def strassen_4x4():
    # We build the 4x4x4 tensor from 2x2x2 tensors.
    # U, V, W will be 16 x 49
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Get 2x2 Strassen tensors
    U2, V2, W2 = strassen_2x2()
    
    # Define the 7 Strassen combinations for 4x4
    # Each combination is a list of (coeff, A_block, B_block)
    # A_block is (row_block, col_block) in 0..1
    # B_block is (row_block, col_block) in 0..1
    
    combos = [
        [ (1, (0,0), (0,0)), (1, (1,1), (1,1)) ], # M1
        [ (1, (1,0), (0,0)), (1, (1,1), (0,0)) ], # M2
        [ (1, (0,0), (0,1)), (-1, (0,0), (1,1)) ], # M3
        [ (1, (1,1), (1,0)), (-1, (1,1), (0,0)) ], # M4
        [ (1, (0,0), (1,1)), (1, (0,1), (1,1)) ], # M5
        [ (1, (1,0), (0,0)), (-1, (0,0), (0,0)), (1, (1,0), (0,1)), (-1, (0,0), (0,1)) ], # M6 (A10-A00)(B00+B01)
        [ (1, (0,1), (1,0)), (-1, (1,1), (1,0)), (1, (0,1), (1,1)), (-1, (1,1), (1,1)) ]  # M7 (A01-A11)(B10+B11)
    ]
    
    # Wait, M6 is (A10 - A00)(B00 + B01).
    # Expansion: A10B00 + A10B01 - A00B00 - A00B01.
    # Each term is a product of a 2x2 block A and a 2x2 block B.
    # We need to sum the Strassen decompositions of these products.
    
    col_offset = 0
    for combo in combos:
        # For each term in the linear combination
        # We need to add the Strassen decomposition of A_block * B_block
        # scaled by the coefficient.
        # But wait, the Strassen decomposition is for the product of the *entire* 2x2 matrices.
        # If we have A_block * B_block, we can use the 2x2 Strassen decomposition directly.
        # The U vector for A_block * B_block will have non-zeros in the rows corresponding to A_block.
        # The V vector for A_block * B_block will have non-zeros in the rows corresponding to B_block.
        # The W vector will be the standard W2.
        
        # However, if we have a sum of such products, we can just sum the tensors.
        # So for each (coeff, A_blk, B_blk) in combo:
        #   For each r in 0..6:
        #       U[global_A_idx][col_offset + r] += coeff * U2[local_A_idx][r]
        #       V[global_B_idx][col_offset + r] += coeff * V2[local_B_idx][r]
        #       W[global_C_idx][col_offset + r] += W2[local_C_idx][r]
        
        # But this is not correct.
        # The Strassen decomposition of X Y is sum u_r v_r w_r.
        # If X = A1, Y = B1, then A1 B1 = sum u_r(A1) v_r(B1) w_r.
        # If we have A1 B1 + A2 B2, we can just concatenate the decompositions.
        # So we can treat each term in the expansion as a separate set of 7 rank-1 tensors.
        
        # But the Strassen algorithm for 4x4 uses 7 *multiplications* of 2x2 matrices.
        # It does NOT expand the sums into products first.
        # It computes M1 = (A00+A11)(B00+B11).
        # This is ONE multiplication of two 2x2 matrices.
        # So we use ONE set of 7 Strassen tensors for M1.
        # The "A" matrix for this multiplication is A00+A11.
        # The "B" matrix for this multiplication is B00+B11.
        
        # So for each of the 7 Strassen steps:
        #   Construct the effective A and B (as linear combinations of blocks).
        #   Apply the 2x2 Strassen decomposition to this effective A and B.
        
        # Effective A is a 2x2 matrix of blocks.
        # But we need U, V, W vectors.
        # U vector will have entries for the blocks involved.
        
        # Let's do this:
        # For each step s (0..6):
        #   Determine the coefficients for A blocks and B blocks.
        #   For each r (0..6) in 2x2 Strassen:
        #       U[global_A][col] = sum(coeff_A_block * U2[local_A_block][r])
        #       V[global_B][col] = sum(coeff_B_block * V2[local_B_block][r])
        #       W[global_C][col] = W2[local_C][r]
        
        # This works!
        
        # We need to define the coefficients for A and B for each step.
        # Step 0: M1 = (A00+A11)(B00+B11)
        # A coeffs: A00:1, A11:1. Others 0.
        # B coeffs: B00:1, B11:1. Others 0.
        
        # We'll represent A coeffs as a dict {(r,c): val}
        
        steps = [
            { (0,0): 1, (1,1): 1 }, # A for M1
            { (0,0): 1, (1,1): 1 }, # B for M1
            { (1,0): 1, (1,1): 1 }, # A for M2
            { (0,0): 1 }, # B for M2
            { (0,0): 1 }, # A for M3
            { (0,1): 1, (1,1): -1 }, # B for M3
            { (1,1): 1 }, # A for M4
            { (1,0): 1, (0,0): -1 }, # B for M4
            { (0,0): 1, (0,1): 1 }, # A for M5
            { (1,1): 1 }, # B for M5
            { (1,0): 1, (0,0): -1 }, # A for M6
            { (0,0): 1, (0,1): 1 }, # B for M6
            { (0,1): 1, (1,1): -1 }, # A for M7
            { (1,0): 1, (1,1): 1 }  # B for M7
        ]
        
        for s in range(7):
            A_coeffs = steps[s*2]
            B_coeffs = steps[s*2 + 1]
            
            for r in range(7):
                col = col_offset + r
                
                # Fill U
                for (br, bc), val in A_coeffs.items():
                    # Local indices in 2x2 block
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2 + lc
                            global_row = (br*2 + lr)*4 + (bc*2 + lc)
                            U[global_row][col] += val * U2[local_idx][r]
                
                # Fill V
                for (br, bc), val in B_coeffs.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr*2 + lc
                            global_row = (br*2 + lr)*4 + (bc*2 + lc)
                            V[global_row][col] += val * V2[local_idx][r]
                
                # Fill W
                # W is the same for all steps, just mapped to global indices
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr*2 + lc
                        # W local index is k*2 + i? No, W2 is for 2x2x2.
                        # W2[k*2+i]
                        # Global W index is k*4 + i.
                        # But W2 is the same for all 7 steps?
                        # Yes, the C part of Strassen 2x2 is fixed.
                        # But we need to add it 7 times?
                        # No, W is the output tensor.
                        # In the decomposition T = sum u v w, the w vector is part of the rank-1 tensor.
                        # For Strassen, the w vectors are the same for all 7 terms?
                        # No, the 7 terms are different rank-1 tensors.
                        # U2, V2, W2 are the 7 rank-1 tensors.
                        # So W[global][col] should be W2[local][r].
                        
                        # Wait, the global W index depends on the global i and k.
                        # The local W index depends on local i and k.
                        # The mapping is direct.
                        
                        global_k = lr
                        global_i = lc
                        # Wait, W2 index is k*2 + i.
                        # So local_k = lr, local_i = lc.
                        # global_k = lr, global_i = lc?
                        # No, the 2x2 block is at (br, bc) for A and B.
                        # The output C is 4x4.
                        # The 2x2 Strassen computes a 2x2 block of C?
                        # No, it computes the whole 4x4 C?
                        # No, Strassen 2x2 computes a 2x2 product.
                        # In Strassen 4x4, each M_k is a 2x2 product.
                        # The result M_k is a 2x2 matrix.
                        # The final C is a linear combination of M_k.
                        # So the W vectors for the rank-1 tensors of M_k should be placed in the correct 2x2 block of C.
                        
                        # Which block of C does M_k contribute to?
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # So for step s, we need to know which C block it goes to.
                        # And with what sign.
                        
                        # Let's define C_coeffs for each step.
                        
                        pass # Will fix in final code
