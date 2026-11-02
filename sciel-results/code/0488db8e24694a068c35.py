import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 98 decomposition.
    # Split p=7 into p1=4 and p2=3 (padded to 4).
    
    # Helper to get Strassen factors for 2x2x2
    # Returns U (4, 7), V (4, 7), W (4, 7)
    def strassen_2x2():
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b00)
        # M4 = a11(b00-b11)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices in flat 2x2: 00->0, 01->1, 10->2, 11->3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[3, 2] = 1; V[0, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[0, 3] = 1; V[3, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1; W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Build 4x4x4 Strassen decomposition
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    u2, v2, w2 = strassen_2x2()
    
    # Macro products M1..M7
    # For each macro product, we need to define the linear combinations of A and B blocks
    # Then map to the 2x2 Strassen structure
    
    # M1: (A11+A22)(B11+B22)
    # A' = A11+A22. B' = B11+B22.
    # A' elements:
    # a00' = A00 + A22
    # a01' = A01 + A23
    # a10' = A10 + A32
    # a11' = A11 + A33
    # B' elements:
    # b00' = B00 + B22
    # b01' = B01 + B23
    # b10' = B10 + B32
    # b11' = B11 + B33
    
    # We iterate over the 7 sub-multiplications of the 2x2 Strassen
    # and for each, we add the contributions to U, V, W.
    
    # Sub-multiplication s (0..6) of u2, v2, w2
    # u2[row, s] is coeff for A' element row
    # v2[row, s] is coeff for B' element row
    # w2[row, s] is coeff for C' element row
    
    # Mapping of 2x2 indices to 4x4 indices
    # A' index 0 (00) -> A11[00] (idx 0) and A22[00] (idx 10? No. A22 is block 1,1. Row 2, Col 2. Flat 2*4+2=10)
    # A11 is block 0,0. Rows 0,1. Cols 0,1.
    # A22 is block 1,1. Rows 2,3. Cols 2,3.
    
    def map_a_block(block_r, block_c, local_r, local_c):
        # block_r, block_c in {0,1}
        # local_r, local_c in {0,1}
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return row * 4 + col
        
    def map_b_block(block_r, block_c, local_r, local_c):
        # B is 4x4.
        # B11 is block 0,0. Rows 0,1. Cols 0,1.
        # B12 is block 0,1. Rows 0,1. Cols 2,3.
        # B21 is block 1,0. Rows 2,3. Cols 0,1.
        # B22 is block 1,1. Rows 2,3. Cols 2,3.
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return row * 4 + col
        
    def map_c_block(block_r, block_c, local_r, local_c):
        # C is 4x4.
        # C11 is block 0,0.
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        # W index: k*n + i. C is k x i.
        # In our C block notation, C11 is rows 0,1 cols 0,1.
        # k is col, i is row.
        return col * 4 + row

    # Define the 7 macro products
    # Each macro product is defined by:
    # A_blocks: list of (block_r, block_c, coeff)
    # B_blocks: list of (block_r, block_c, coeff)
    # C_blocks: list of (block_r, block_c, coeff) -- actually C is determined by the product
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B22-B11)
    # M4: A22(B11-B22)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    macros = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: dict { (block_r, block_c): coeff }
        # B_coeffs: dict { (block_r, block_c): coeff }
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}), # M2
        ({(0,0): 1}, {(1,1): 1, (0,0): -1}), # M3
        ({(1,1): 1}, {(0,0): 1, (1,1): -1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}), # M7
    ]
    
    for m_idx, (a_coeffs, b_coeffs) in enumerate(macros):
        # For each of the 7 sub-multiplications in 2x2 Strassen
        for s_idx in range(7):
            col_idx = m_idx * 7 + s_idx
            
            # Get u2, v2, w2 coefficients for this sub-multiplication
            # u2 is 4x7. Row is A' local index.
            # A' local index 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
            
            # Add to U
            for local_r in range(2):
                for local_c in range(2):
                    coeff_u = u2[local_r*2+local_c, s_idx]
                    if coeff_u != 0:
                        # Find which A blocks contribute to this local position of A'
                        # A' = sum_{block} coeff * A_block
                        # The local position (local_r, local_c) in A' corresponds to
                        # the same local position in each A_block.
                        for (br, bc), coeff_a in a_coeffs.items():
                            if coeff_a != 0:
                                global_idx = map_a_block(br, bc, local_r, local_c)
                                U_4x4[global_idx, col_idx] += coeff_u * coeff_a
                                
            # Add to V
            for local_r in range(2):
                for local_c in range(2):
                    coeff_v = v2[local_r*2+local_c, s_idx]
                    if coeff_v != 0:
                        for (br, bc), coeff_b in b_coeffs.items():
                            if coeff_b != 0:
                                global_idx = map_b_block(br, bc, local_r, local_c)
                                V_4x4[global_idx, col_idx] += coeff_v * coeff_b
                                
            # Add to W
            # C' = A' B'.
            # The result C' is 2x2.
            # We need to add C' to the appropriate C blocks.
            # C' local (lr, lc) contributes to C block (br, bc) with coeff 1?
            # Yes, C = sum C'_block?
            # No, C is the result of the macro product.
            # The macro product result is a 2x2 matrix.
            # This 2x2 matrix is added to specific C blocks.
            # For M1, the result is added to C11, C12, C21, C22?
            # Wait. M1 = (A11+A22)(B11+B22).
            # C11 = M1[00] + ...
            # C12 = M1[01] + ...
            # C21 = M1[10] + ...
            # C22 = M1[11] + ...
            # Actually, in Strassen:
            # C11 = M1 + M4 - M3 - M7
            # C12 = M5 + M7
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for each macro product m_idx, we know which C blocks it contributes to.
            
            # Let's define the C contributions for each macro.
            # M1 contributes to C11, C12, C21, C22.
            # Specifically:
            # M1[00] -> C11[00], C12[00]? No.
            # M1 is a 2x2 matrix.
            # M1[00] adds to C11[00]?
            # Let's check the formula.
            # C11 = M1 + M4 - M3 - M7.
            # This means the entire 2x2 matrix M1 is added to C11?
            # No. M1 is 2x2. C11 is 2x2.
            # So M1[00] adds to C11[00], M1[01] adds to C11[01], etc.
            # So M1 contributes to C11, C12, C21, C22?
            # Let's re-verify Strassen.
            # C11 = M1 + M4 - M3 - M7
            # C12 = M5 + M7
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So M1 contributes to C11 and C22.
            # M2 contributes to C21 and C22.
            # M3 contributes to C11 and C22.
            # M4 contributes to C11 and C21.
            # M5 contributes to C12.
            # M6 contributes to C22.
            # M7 contributes to C11, C12.
            
            c_contributions = [
                [(0,0), (1,1)], # M1
                [(1,0), (1,1)], # M2
                [(0,0), (1,1)], # M3
                [(0,0), (1,0)], # M4
                [(0,1)], # M5
                [(1,1)], # M6
                [(0,0), (0,1)], # M7
            ]
            
            for local_r in range(2):
                for local_c in range(2):
                    coeff_w = w2[local_r*2+local_c, s_idx]
                    if coeff_w != 0:
                        for (br, bc) in c_contributions[m_idx]:
                            global_idx = map_c_block(br, bc, local_r, local_c)
                            W_4x4[global_idx, col_idx] += coeff_w

    # Now we have U_4x4, V_4x4, W_4x4 for the first 4 columns.
    # For the second 3 columns (padded to 4), we use the same U and W.
    # V is different.
    
    # Construct V_4x4_B
    # B_B blocks:
    # B_B11: cols 4,5 (0-indexed 4,5? No. B is 4x7. Cols 0..6.
    # B_A uses cols 0,1,2,3.
    # B_B uses cols 4,5,6,7(0).
    # So B_B11 uses cols 4,5.
    # B_B12 uses cols 6,7(0).
    # B_B21 uses cols 4,5.
    # B_B22 uses cols 6,7(0).
    
    V_4x4_B = np.zeros((16, 49))
    
    # We can reuse the logic, but with different B block mappings.
    # Or just copy V_4x4 and modify the columns corresponding to B.
    # Actually, the structure of V depends on which B blocks are non-zero.
    # In V_4x4, the entries are non-zero for B11, B12, B21, B22.
    # In V_4x4_B, B11 is at cols 4,5. B12 is at cols 6,7.
    # B21 is at cols 4,5. B22 is at cols 6,7.
    
    # Let's just rebuild V_4x4_B using the same macro logic but mapping to different global indices.
    
    def map_b_block_B(block_r, block_c, local_r, local_c):
        # B_B is 4x4.
        # Block 0,0 (B11) -> Rows 0,1. Cols 4,5.
        # Block 0,1 (B12) -> Rows 0,1. Cols 6,7(0).
        # Block 1,0 (B21) -> Rows 2,3. Cols 4,5.
        # Block 1,1 (B22) -> Rows 2,3. Cols 6,7(0).
        
        if block_c == 0:
            col = 4 + local_c
        else:
            col = 6 + local_c # 6, 7(0)
            
        row = block_r * 2 + local_r
        return row * 4 + col
        
    # Rebuild V_4x4_B
    for m_idx, (a_coeffs, b_coeffs) in enumerate(macros):
        for s_idx in range(7):
            col_idx = m_idx * 7 + s_idx
            for local_r in range(2):
                for local_c in range(2):
                    coeff_v = v2[local_r*2+local_c, s_idx]
                    if coeff_v != 0:
                        for (br, bc), coeff_b in b_coeffs.items():
                            if coeff_b != 0:
                                global_idx = map_b_block_B(br, bc, local_r, local_c)
                                V_4x4_B[global_idx, col_idx] += coeff_v * coeff_b

    # Combine
    U = np.hstack([U_4x4, U_4x4])
    V = np.hstack([V_4x4, V_4x4_B])
    W = np.hstack([W_4x4, W_4x4])
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

