import numpy as np

DIMENSIONS = [4, 4, 6]

def get_222_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    """
    # The 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1 # C12, C11
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1 # C11, C22
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We are doing 4x4x6
    # Split into 4x4x4 and 4x4x2
    
    # Base decomposition
    U2, V2, W2 = get_222_strassen()
    
    # Part 1: 4x4x4 (Rank 49)
    # 7 high-level terms
    # Each term uses the 2x2x2 decomposition
    # We need to map local indices to global indices
    
    U_part1 = np.zeros((16, 49))
    V_part1 = np.zeros((24, 49)) # 4x6, but we only use first 4 cols for B1? 
    # Wait, B is 4x6. B1 is 4x4. B2 is 4x2.
    # For Part 1, B indices are 0..3 (cols 0-3).
    # For Part 2, B indices are 4..5 (cols 4-5).
    
    # High level terms definitions
    # (A_blocks, B_blocks, C_blocks, signs)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Term 1: (A00+A11)(B00+B11) -> C00(+), C11(+)
    # Term 2: (A10+A11)B00 -> C10(+)
    # Term 3: A00(B01-B11) -> C01(+)
    # Term 4: A11(B10-B00) -> C00(+), C10(+)
    # Term 5: (A00+A01)B11 -> C01(+), C00(-)
    # Term 6: (A10-A00)(B00+B01) -> C11(+)
    # Term 7: (A01-A11)(B10+B11) -> C00(+), C11(-)
    
    # Note: My indices in get_222_strassen were:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Let's define the mappings for the 7 terms
    # Each term k has:
    # a_map: dict local_a_idx -> list of (global_a_idx, coeff)
    # b_map: dict local_b_idx -> list of (global_b_idx, coeff)
    # c_map: dict local_c_idx -> list of (global_c_idx, coeff)
    
    # Global A indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Global B indices (for Part 1, cols 0-3):
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # Global C indices:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Helper to get global indices for a block
    def get_block_indices(block_id, total_cols=4):
        # block_id: 00, 01, 10, 11
        r = int(block_id[0])
        c = int(block_id[1])
        start_r = r * 2
        start_c = c * 2
        idxs = []
        for i in range(2):
            for j in range(2):
                idxs.append((start_r + i) * total_cols + (start_c + j))
        return idxs
    
    # For B in Part 1, total_cols is 4 (local to B1)
    # But global B has 6 cols.
    # B1 uses cols 0-3. So global B indices are same as local B1 indices.
    # B2 uses cols 4-5.
    
    # We will build U, V, W for Part 1
    col_offset = 0
    for k in range(7):
        # Determine mappings for term k
        # Using the logic from get_222_strassen
        # We need to map local 2x2 blocks to global 4x4 blocks
        
        # Local A index 0 (00) maps to which global blocks?
        # This depends on the term.
        
        # Let's hardcode the block combinations for the 7 Strassen terms
        # Term 0: A: 00,11; B: 00,11; C: 00,11 (all +1)
        # Term 1: A: 10,11; B: 00; C: 10 (+1)
        # Term 2: A: 00; B: 01,11; C: 01 (+1 for 01, -1 for 11)
        # Term 3: A: 11; B: 10,00; C: 00,10 (+1 for 10, -1 for 00) -> Wait, M4 = A11(B10-B00)
        # Term 4: A: 00,01; B: 11; C: 01,00 (+1 for 01, -1 for 00) -> M5 = (A00+A01)B11
        # Term 5: A: 10,00; B: 00,01; C: 11 (+1 for 10, -1 for 00; +1 for 00, +1 for 01) -> M6 = (A10-A00)(B00+B01)
        # Term 6: A: 01,11; B: 10,11; C: 00,11 (+1 for 01, -1 for 11; +1 for 10, +1 for 11) -> M7 = (A01-A11)(B10+B11)
        
        # We need to be careful with signs.
        
        # Let's use the U2, V2, W2 matrices directly.
        # For each local element in U2, we map it to global elements.
        
        # Define block mappings for each term
        # term_k: (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (local_block, coeff)
        
        mappings = [
            # 0
            {'a': [('00', 1), ('11', 1)], 'b': [('00', 1), ('11', 1)], 'c': [('00', 1), ('11', 1)]},
            # 1
            {'a': [('10', 1), ('11', 1)], 'b': [('00', 1)], 'c': [('10', 1)]},
            # 2
            {'a': [('00', 1)], 'b': [('01', 1), ('11', -1)], 'c': [('01', 1)]},
            # 3
            {'a': [('11', 1)], 'b': [('10', 1), ('00', -1)], 'c': [('00', 1), ('10', 1)]},
            # 4
            {'a': [('00', 1), ('01', 1)], 'b': [('11', 1)], 'c': [('01', 1), ('00', -1)]},
            # 5
            {'a': [('10', 1), ('00', -1)], 'b': [('00', 1), ('01', 1)], 'c': [('11', 1)]},
            # 6
            {'a': [('01', 1), ('11', -1)], 'b': [('10', 1), ('11', 1)], 'c': [('00', 1), ('11', -1)]},
        ]
        
        for l in range(4): # Local index in 2x2
            # U2[l, k] is the coefficient
            val_u = U2[l, k]
            if val_u != 0:
                # Map local l to global A indices
                # l is 0,1,2,3 corresponding to 00,01,10,11
                # But U2 is flattened.
                # U2[0] is A00, U2[1] is A01, U2[2] is A10, U2[3] is A11
                # Wait, in get_222_strassen, I used:
                # U[0,0]=1 (A00)
                # U[3,0]=1 (A11)
                # So l=0 is A00, l=1 is A01, l=2 is A10, l=3 is A11
                local_block = ['00', '01', '10', '11'][l]
                for gb, coeff in mappings[k]['a']:
                    if gb == local_block:
                        # Add val_u * coeff to global A indices
                        g_idxs = get_block_indices(gb, 4)
                        for g_idx in g_idxs:
                            U_part1[g_idx, col_offset + l] = val_u * coeff # This is wrong. 
                            # The sub-decomposition U2[l, k] multiplies the local A element.
                            # The local A element is a sum of global blocks.
                            # So U_global[global_idx, col] += U2[l, k] * coeff
                            # But U2 is for the sub-problem.
                            # The column in U_part1 corresponds to a specific rank-1 term in the sub-problem.
                            # Let's say we are at sub-column c_sub.
                            # The global column is col_offset + c_sub.
                            # U_part1[global_idx, global_col] += val_u * coeff
                            pass
            # This loop is getting messy. 
            # Better to iterate over sub-columns.
            
        # Let's redo the combination logic cleanly.
        # For term k, we have sub-decomposition U2[:, k], V2[:, k], W2[:, k].
        # These are vectors of length 4.
        # They represent the coefficients for the 4 local elements.
        # We need to expand these into global coefficients.
        
        for c_sub in range(7): # 7 sub-columns
            global_col = col_offset + c_sub
            
            # Process U
            for l in range(4):
                val = U2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in mappings[k]['a']:
                        if gb == local_block:
                            g_idxs = get_block_indices(gb, 4)
                            for g_idx in g_idxs:
                                U_part1[g_idx, global_col] += val * coeff
            
            # Process V
            for l in range(4):
                val = V2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in mappings[k]['b']:
                        if gb == local_block:
                            g_idxs = get_block_indices(gb, 4)
                            for g_idx in g_idxs:
                                V_part1[g_idx, global_col] += val * coeff
            
            # Process W
            for l in range(4):
                val = W2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in mappings[k]['c']:
                        if gb == local_block:
                            g_idxs = get_block_indices(gb, 4)
                            for g_idx in g_idxs:
                                W_part1[g_idx, global_col] += val * coeff
                            
        col_offset += 7
        
    # Part 1 done. Rank 49.
    
    # Part 2: 4x4x2 (Rank 28)
    # B2 is cols 4,5.
    # Split B2 into B2_top (2x2) and B2_bot (2x2).
    # B2_top corresponds to global B indices:
    # Row 0,1; Col 4,5 -> 4,5,8,9
    # B2_bot corresponds to global B indices:
    # Row 2,3; Col 4,5 -> 12,13,16,17
    
    # 4 terms:
    # 1. A00 * B2_top -> C00
    # 2. A01 * B2_bot -> C00
    # 3. A10 * B2_top -> C10
    # 4. A11 * B2_bot -> C10
    
    U_part2 = np.zeros((16, 28))
    V_part2 = np.zeros((24, 28))
    W_part2 = np.zeros((24, 28)) # C is 4x6, so 24 elements.
    
    col_offset2 = 0
    for t in range(4):
        # Define mappings
        if t == 0:
            a_map = [('00', 1)]
            b_map = [('top', 1)]
            c_map = [('00', 1)]
        elif t == 1:
            a_map = [('01', 1)]
            b_map = [('bot', 1)]
            c_map = [('00', 1)]
        elif t == 2:
            a_map = [('10', 1)]
            b_map = [('top', 1)]
            c_map = [('10', 1)]
        else:
            a_map = [('11', 1)]
            b_map = [('bot', 1)]
            c_map = [('10', 1)]
            
        for c_sub in range(7):
            global_col = col_offset2 + c_sub
            
            for l in range(4):
                val = U2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in a_map:
                        if gb == local_block:
                            g_idxs = get_block_indices(gb, 4)
                            for g_idx in g_idxs:
                                U_part2[g_idx, global_col] += val * coeff
                                
            for l in range(4):
                val = V2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in b_map:
                        if gb == 'top':
                            g_idxs = [4,5,8,9]
                        else:
                            g_idxs = [12,13,16,17]
                        # The local B block in the sub-problem is 2x2.
                        # We are mapping the local block to the global block.
                        # The indices in get_block_indices are for 4x4.
                        # Here we need specific indices.
                        for g_idx in g_idxs:
                            V_part2[g_idx, global_col] += val * coeff
                            
            for l in range(4):
                val = W2[l, c_sub]
                if val != 0:
                    local_block = ['00', '01', '10', '11'][l]
                    for gb, coeff in c_map:
                        if gb == '00':
                            g_idxs = get_block_indices('00', 6) # C is 4x6
                        else:
                            g_idxs = get_block_indices('10', 6)
                        for g_idx in g_idxs:
                            W_part2[g_idx, global_col] += val * coeff
                            
        col_offset2 += 7
        
    # Combine
    U = np.hstack([U_part1, U_part2])
    V = np.hstack([V_part1, V_part2])
    W = np.hstack([W_part1, W_part2])
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
