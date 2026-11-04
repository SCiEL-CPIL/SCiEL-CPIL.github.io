DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))

    # --- Part 1: 4x4x4 using Strassen ---
    
    # Define 2x2 Strassen decomposition terms
    # Returns list of (u_2x2, v_2x2, w_2x2) for 2x2x2 multiplication
    def strassen_2x2_terms():
        # u, v, w are 4-element lists (flattened 2x2)
        # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        t = []
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C00, C11
        t.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # C10, C11
        t.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1] # C01, C11
        t.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # C00, C10
        t.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # C00, C01
        t.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11
        t.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00
        t.append((u, v, w))
        
        return t

    # The 7 block multiplications for 4x4
    # Each is defined by linear forms on A and B blocks
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # We need to map the 2x2 local indices to global 4x4 indices
    # Global A is 4x4.
    # A00: rows 0-1, cols 0-1. Indices 0,1,4,5
    # A01: rows 0-1, cols 2-3. Indices 2,3,6,7
    # A10: rows 2-3, cols 0-1. Indices 8,9,12,13
    # A11: rows 2-3, cols 2-3. Indices 10,11,14,15
    
    # Global B is 4x4 (for the first part).
    # B00: rows 0-1, cols 0-1. Indices 0,1,4,5
    # B01: rows 0-1, cols 2-3. Indices 2,3,6,7
    # B10: rows 2-3, cols 0-1. Indices 8,9,12,13
    # B11: rows 2-3, cols 2-3. Indices 10,11,14,15
    
    # Global C is 4x4.
    # C00: rows 0-1, cols 0-1. Indices 0,1,4,5
    # C01: rows 0-1, cols 2-3. Indices 2,3,6,7
    # C10: rows 2-3, cols 0-1. Indices 8,9,12,13
    # C11: rows 2-3, cols 2-3. Indices 10,11,14,15

    # Function to expand a 2x2 local term to 4x4 global
    def expand_2x2_to_4x4(local_u, local_v, local_w, 
                           a_map, b_map, c_map):
        # local_u, local_v, local_w are length 4
        # a_map: list of 4 global indices for the 2x2 block in A
        # b_map: list of 4 global indices for the 2x2 block in B
        # c_map: list of 4 global indices for the 2x2 block in C
        
        global_u = [0] * 16
        global_v = [0] * 16
        global_w = [0] * 16
        
        for i in range(4):
            global_u[a_map[i]] += local_u[i]
            global_v[b_map[i]] += local_v[i]
            global_w[c_map[i]] += local_w[i]
            
        return global_u, global_v, global_w

    # Define the 7 Strassen steps for 4x4
    # Each step: (U_lin, V_lin, C_block)
    # U_lin is a 2x2 matrix of coefficients for A blocks
    # V_lin is a 2x2 matrix of coefficients for B blocks
    
    # We represent the linear combination of 2x2 blocks as a list of 4 (block_index, coeff)
    # Or simpler: just define the 2x2 "super-matrix" of coefficients.
    
    # M1 = (A00 + A11)(B00 + B11)
    # X1 = A00 + A11. Y1 = B00 + B11.
    # The 2x2 Strassen terms are applied to X1 and Y1.
    # So for each of the 7 local terms (lu, lv, lw):
    # global_u gets lu expanded over A00 and A11.
    # global_v gets lv expanded over B00 and B11.
    # global_w gets lw expanded over C00.
    
    # Let's define the mappings
    # A00 map: [0, 1, 4, 5]
    # A01 map: [2, 3, 6, 7]
    # A10 map: [8, 9, 12, 13]
    # A11 map: [10, 11, 14, 15]
    
    # B00 map: [0, 1, 4, 5]
    # B01 map: [2, 3, 6, 7]
    # B10 map: [8, 9, 12, 13]
    # B11 map: [10, 11, 14, 15]
    
    # C00 map: [0, 1, 4, 5]
    # C01 map: [2, 3, 6, 7]
    # C10 map: [8, 9, 12, 13]
    # C11 map: [10, 11, 14, 15]

    a_maps = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15]# A11
    }
    
    b_maps = {
        0: [0, 1, 4, 5],   # B00
        1: [2, 3, 6, 7],   # B01
        2: [8, 9, 12, 13], # B10
        3: [10, 11, 14, 15]# B11
    }
    
    c_maps = {
        0: [0, 1, 4, 5],   # C00
        1: [2, 3, 6, 7],   # C01
        2: [8, 9, 12, 13], # C10
        3: [10, 11, 14, 15]# C11
    }

    strassen_2x2 = strassen_2x2_terms()
    
    # Define the 7 combinations
    # Format: (list of (a_block, coeff), list of (b_block, coeff), c_block)
    # We need to sum the contributions to global_u for each a_block.
    
    def process_strassen_step(a_blocks, b_blocks, c_block):
        for lu, lv, lw in strassen_2x2:
            gu = [0] * 16
            gv = [0] * 16
            gw = [0] * 16
            
            for block_idx, coeff in a_blocks:
                if coeff != 0:
                    map_ = a_maps[block_idx]
                    for i in range(4):
                        gu[map_[i]] += lu[i] * coeff
                        
            for block_idx, coeff in b_blocks:
                if coeff != 0:
                    map_ = b_maps[block_idx]
                    for i in range(4):
                        gv[map_[i]] += lv[i] * coeff
                        
            map_ = c_maps[c_block]
            for i in range(4):
                gw[map_[i]] += lw[i]
                
            add_term(gu, gv, gw)

    # M1: (A00+A11)(B00+B11) -> C00
    process_strassen_step([(0, 1), (3, 1)], [(0, 1), (3, 1)], 0)
    
    # M2: (A10+A11)B00 -> C10, C11
    # Note: The output of M2 is a 2x2 matrix.
    # In the 4x4 formula:
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M2 contributes to C10 and C11.
    # But my process_strassen_step only maps to ONE c_block.
    # I need to modify it to handle multiple c_blocks or call it twice?
    # No, the 2x2 Strassen decomposition produces a 2x2 output matrix.
    # The local w vector (lw) has 4 entries corresponding to the 4 entries of the 2x2 output.
    # So I need to map the 4 entries of lw to the correct global C indices.
    
    # Let's redefine process_strassen_step to take a list of c_blocks and how to map the 2x2 output.
    # Actually, the 2x2 output of M2 corresponds to the top-left 2x2 of C10? No.
    # M2 is a 2x2 matrix.
    # C10 is a 2x2 matrix. C11 is a 2x2 matrix.
    # The formula is C10 = M2 + M4.
    # This means the entire 2x2 matrix M2 is added to C10.
    # And the entire 2x2 matrix M2 is subtracted from C11?
    # No, the formula is for the blocks.
    # C10 (block) = M2 (block) + M4 (block).
    # So the 4 entries of M2 are added to the 4 entries of C10.
    # And the 4 entries of M2 are subtracted from the 4 entries of C11?
    # Let's check the 2x2 Strassen output mapping.
    # In 2x2 Strassen, the output C is 2x2.
    # So if we compute M2 = X2 Y2, the result is a 2x2 matrix.
    # This 2x2 matrix IS the block C10? No, C10 is a block of C.
    # Wait, in 4x4 Strassen, M2 is a 2x2 matrix.
    # C10 is a 2x2 matrix.
    # The equation C10 = M2 + M4 implies that the 2x2 matrix M2 is added to the 2x2 matrix C10.
    # So the 4 entries of M2 map to the 4 entries of C10.
    # Does M2 also map to C11?
    # C11 = M1 - M2 + M3 + M6.
    # Yes, the 2x2 matrix M2 is subtracted from C11.
    # So I need to add the contribution of M2 to both C10 and C11.
    
    # My current process_strassen_step only adds to one c_block.
    # I will create a new function that takes a list of (c_block, coeff) for the output.

    def process_strassen_step_multi(a_blocks, b_blocks, c_targets):
        for lu, lv, lw in strassen_2x2:
            gu = [0] * 16
            gv = [0] * 16
            gw = [0] * 16
            
            for block_idx, coeff in a_blocks:
                if coeff != 0:
                    map_ = a_maps[block_idx]
                    for i in range(4):
                        gu[map_[i]] += lu[i] * coeff
                        
            for block_idx, coeff in b_blocks:
                if coeff != 0:
                    map_ = b_maps[block_idx]
                    for i in range(4):
                        gv[map_[i]] += lv[i] * coeff
                        
            for c_block, coeff in c_targets:
                if coeff != 0:
                    map_ = c_maps[c_block]
                    for i in range(4):
                        gw[map_[i]] += lw[i] * coeff
                        
            add_term(gu, gv, gw)

    # M1: (A00+A11)(B00+B11)
    # C00 = M1 + ...
    # C11 = M1 + ...
    process_strassen_step_multi([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    
    # M2: (A10+A11)B00
    # C10 = M2 + ...
    # C11 = ... - M2 ...
    process_strassen_step_multi([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    
    # M3: A00(B01-B11)
    # C01 = M3 + ...
    # C11 = ... + M3 ...
    process_strassen_step_multi([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    
    # M4: A11(B10-B00)
    # C00 = ... + M4 ...
    # C10 = M4 + ...
    process_strassen_step_multi([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    
    # M5: (A00+A01)B11
    # C00 = ... - M5 ...
    # C01 = M5 + ...
    process_strassen_step_multi([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
    
    # M6: (A10-A00)(B00+B01)
    # C11 = ... + M6 ...
    process_strassen_step_multi([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # M7: (A01-A11)(B10+B11)
    # C00 = ... + M7 ...
    process_strassen_step_multi([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    
    # --- Part 2: 4x4x3 Schoolbook ---
    # B2 is columns 4-6 of B (indices 4,5,6 in p dimension? No, p=7. Indices 0-6.
    # We used B0-B3 for the 4x4 part.
    # So B4, B5, B6 are the remaining 3 columns.
    # A is the same 4x4.
    # C4, C5, C6 are the output columns.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = [0] * 16
                v = [0] * 28
                w = [0] * 28
                
                u[i * 4 + j] = 1
                v[j * 7 + (4 + k)] = 1
                w[(4 + k) * 4 + i] = 1
                
                add_term(u, v, w)
                
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(16):
            U[r][c] = u[r]
        for r in range(28):
            V[r][c] = v[r]
        for r in range(28):
            W[r][c] = w[r]
            
    return {"U": U, "V": V, "W": W}
