import numpy as np

def get_strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # U is 4x7, V is 4x7, W is 4x7
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # Flattened indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    # U col 0: A11=1, A22=1
    U[0, 0] = 1; U[3, 0] = 1
    # V col 0: B11=1, B22=1
    V[0, 0] = 1; V[3, 0] = 1
    # W col 0: C11+=1, C22+=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C21+=M2, C22+=M2? No.
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 contributes to C21 and C22.
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12+=M3, C22+=M3
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11+=M4, C21+=M4
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11-=M5, C12+=M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22+=M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11+=M7
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # We build this by applying Strassen to 2x2 blocks.
    # The 2x2x2 decomposition has rank 7.
    # The block-level Strassen has 7 multiplications.
    # So total rank is 7 * 7 = 49.
    
    U_222, V_222, W_222 = get_strassen_222()
    
    # Block level Strassen formulas for 4x4x4
    # A is 2x2 blocks: A11, A12, A21, A22
    # B is 2x2 blocks: B11, B12, B21, B22
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix product.
    # We need to map the global indices (0..15) to local indices (0..3) for each block.
    
    # Global A indices:
    # 0,1 -> A11 (0,0; 0,1)
    # 2,3 -> A12 (0,2; 0,3)
    # 4,5 -> A21 (1,0; 1,1)
    # 6,7 -> A22 (1,2; 1,3)
    # Wait, standard row-major flattening of 4x4:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    
    # Blocks:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Block A11 local indices:
    # 0->0, 1->1, 4->2, 5->3
    
    def block_indices(block):
        if block == 0: # A11
            return [0, 1, 4, 5]
        elif block == 1: # A12
            return [2, 3, 6, 7]
        elif block == 2: # A21
            return [8, 9, 12, 13]
        elif block == 3: # A22
            return [10, 11, 14, 15]
            
    # Same for B and C.
    
    # We will construct U_444, V_444, W_444 of size 16x49.
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Define the 7 block-level multiplications
    # Each is a tuple (A_blocks, coeffs_A, B_blocks, coeffs_B, C_blocks, coeffs_C)
    # Actually, it's simpler to define the linear combinations for A and B inputs
    # and the output C contributions.
    
    # M1: A = A11 + A22, B = B11 + B22, C = C11 + C22
    # M2: A = A21 + A22, B = B11, C = C21 - C22 (Wait, C21+=M2, C22-=M2)
    # M3: A = A11, B = B12 - B22, C = C12 + C22
    # M4: A = A22, B = B21 - B11, C = C11 + C21
    # M5: A = A11 + A12, B = B22, C = -C11 + C12
    # M6: A = A21 - A11, B = B11 + B12, C = C22
    # M7: A = A12 - A22, B = B21 + B22, C = C11
    
    block_defs = [
        # (A_block_indices, A_coeffs, B_block_indices, B_coeffs, C_block_indices, C_coeffs)
        # C_coeffs are added to the corresponding C blocks.
        # Note: The C contribution is distributed to the 2x2 blocks of C.
        # The inner Strassen decomposition handles the 2x2 structure of the result.
        
        # M1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for c_idx, (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs) in enumerate(block_defs):
        # For each of the 7 inner rank-1 terms
        for inner_idx in range(7):
            global_col = c_idx * 7 + inner_idx
            
            # Combine U
            for b_idx, coeff in zip(a_blocks, a_coeffs):
                local_indices = block_indices(b_idx)
                for l_idx, g_idx in enumerate(local_indices):
                    U_444[g_idx, global_col] += coeff * U_222[l_idx, inner_idx]
            
            # Combine V
            for b_idx, coeff in zip(b_blocks, b_coeffs):
                local_indices = block_indices(b_idx)
                for l_idx, g_idx in enumerate(local_indices):
                    V_444[g_idx, global_col] += coeff * V_222[l_idx, inner_idx]
            
            # Combine W
            for b_idx, coeff in zip(c_blocks, c_coeffs):
                local_indices = block_indices(b_idx)
                for l_idx, g_idx in enumerate(local_indices):
                    W_444[g_idx, global_col] += coeff * W_222[l_idx, inner_idx]
                    
    # Part 2: 4x4x2 decomposition (Rank 28)
    # We need 4 independent 2x2x2 multiplications.
    # A blocks: A11, A12, A21, A22
    # B blocks: B_top (2x2), B_bot (2x2)
    # C blocks: C_top (2x2), C_bot (2x2)
    
    # Multiplications:
    # 1. A11 * B_top -> C_top
    # 2. A12 * B_bot -> C_top
    # 3. A21 * B_top -> C_bot
    # 4. A22 * B_bot -> C_bot
    
    # We append these to U, V, W.
    # Total rank so far: 49.
    # Add 4 * 7 = 28 columns.
    # Total rank 77.
    
    U_final = np.zeros((16, 77), dtype=int)
    V_final = np.zeros((16, 77), dtype=int)
    # W for 4x4x6 has size 24 (6*4)
    W_final = np.zeros((24, 77), dtype=int)
    
    # Copy Part 1
    U_final[:, :49] = U_444
    V_final[:, :49] = V_444
    # W_444 is 16x49 (for 4x4 output).
    # We need to map the 4x4 output indices to the first 4 columns of the 4x6 output.
    # Output C is 4x6.
    # Indices 0..15 correspond to C[:, 0:4] if we flatten row-major?
    # Task says: W[k*n+i, r].
    # n=4, p=6.
    # W index is k*n + i.
    # k is column index (0..5), i is row index (0..3).
    # So W[0..3] is col 0, W[4..7] is col 1, etc.
    # The 4x4 output corresponds to columns 0,1,2,3.
    # So we just copy W_444 to W_final[:, :49].
    W_final[:, :49] = W_444
    
    # Now add Part 2
    # Multiplication 1: A11 * B_top -> C_top
    # A11 global indices: 0,1,4,5
    # B_top: B is 4x6. Top half is rows 0,1. Cols 4,5?
    # Wait, B is 4x6.
    # We split B into B1 (4x4) and B2 (4x2).
    # B1 corresponds to cols 0..3.
    # B2 corresponds to cols 4..5.
    # In Part 1, we computed C1 = A * B1 (4x4).
    # In Part 2, we compute C2 = A * B2 (4x2).
    
    # B2 is 4x2.
    # Split B2 into B2_top (2x2) and B2_bot (2x2).
    # B2 rows 0,1 -> B2_top.
    # B2 rows 2,3 -> B2_bot.
    # B2 cols 0,1 -> global cols 4,5.
    
    # Global B indices for B2:
    # Row 0: 0*6+4, 0*6+5 -> 4, 5
    # Row 1: 1*6+4, 1*6+5 -> 10, 11
    # Row 2: 2*6+4, 2*6+5 -> 16, 17
    # Row 3: 3*6+4, 3*6+5 -> 22, 23
    
    # B2_top (rows 0,1 of B2): indices 4,5,10,11
    # B2_bot (rows 2,3 of B2): indices 16,17,22,23
    
    # C2 is 4x2.
    # C2_top (rows 0,1 of C2): global W indices for cols 4,5, rows 0,1.
    # W index = k*n + i.
    # Col 4 (k=4), Row 0 (i=0) -> 4*4+0 = 16
    # Col 5 (k=5), Row 0 (i=0) -> 5*4+0 = 20
    # Col 4, Row 1 (i=1) -> 17
    # Col 5, Row 1 (i=1) -> 21
    # So C2_top W indices: 16,17,20,21
    
    # C2_bot (rows 2,3 of C2):
    # Col 4, Row 2 (i=2) -> 18
    # Col 5, Row 2 (i=2) -> 22
    # Col 4, Row 3 (i=3) -> 19
    # Col 5, Row 3 (i=3) -> 23
    # So C2_bot W indices: 18,19,22,23
    
    # Multiplications for Part 2:
    # 1. A11 * B2_top -> C2_top
    # 2. A12 * B2_bot -> C2_top
    # 3. A21 * B2_top -> C2_bot
    # 4. A22 * B2_bot -> C2_bot
    
    part2_defs = [
        # (A_block, B_block_indices, C_block_indices)
        # A_block is 0,1,2,3 corresponding to A11,A12,A21,A22
        # B_block_indices are global indices of the 2x2 block in B
        # C_block_indices are global indices of the 2x2 block in W
        (0, [4, 5, 10, 11], [16, 17, 20, 21]), # A11 * B2_top -> C2_top
        (1, [16, 17, 22, 23], [16, 17, 20, 21]), # A12 * B2_bot -> C2_top
        (2, [4, 5, 10, 11], [18, 19, 22, 23]), # A21 * B2_top -> C2_bot
        (3, [16, 17, 22, 23], [18, 19, 22, 23]), # A22 * B2_bot -> C2_bot
    ]
    
    col_offset = 49
    for p_idx, (a_block, b_indices, c_indices) in enumerate(part2_defs):
        for inner_idx in range(7):
            global_col = col_offset + p_idx * 7 + inner_idx
            
            # U
            a_local_indices = block_indices(a_block)
            for l_idx, g_idx in enumerate(a_local_indices):
                U_final[g_idx, global_col] = U_222[l_idx, inner_idx]
            
            # V
            for l_idx, g_idx in enumerate(b_indices):
                V_final[g_idx, global_col] = V_222[l_idx, inner_idx]
            
            # W
            for l_idx, g_idx in enumerate(c_indices):
                W_final[g_idx, global_col] = W_222[l_idx, inner_idx]
                
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
