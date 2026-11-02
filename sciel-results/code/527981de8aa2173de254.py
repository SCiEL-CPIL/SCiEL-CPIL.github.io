import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 110
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if coeff != 0:
            U[u_idx, col] = coeff
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # 1. A44 B44 (49 terms) using Strassen
    # We need to implement the 4x4 Strassen decomposition.
    
    # 2x2 Strassen factors
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[1, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # 4x4 Strassen structure
    # Blocks: 00, 01, 10, 11
    # Indices in 4x4 matrix:
    # 00: 0,1 x 0,1
    # 01: 0,1 x 2,3
    # 10: 2,3 x 0,1
    # 11: 2,3 x 2,3
    
    # We define the 7 multiplications M1..M7 for 4x4
    # Each is X * Y
    # X and Y are linear combos of A_blocks and B_blocks
    # We store coefficients for A_blocks and B_blocks
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output C blocks:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mk, we iterate over the 7 Strassen terms of 2x2
    # and add them to global U, V, W
    
    # Block definitions: (row_start, row_end, col_start, col_end)
    blocks = {
        0: (0, 2, 0, 2), # 00
        1: (0, 2, 2, 4), # 01
        2: (2, 4, 0, 2), # 10
        3: (2, 4, 2, 4)  # 11
    }
    
    # Coefficients for X and Y for each M
    # M: (coeffs_X, coeffs_Y)
    # coeffs is dict {block_id: coeff}
    M_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Coefficients for C blocks from each M
    # C_block: {M_id: coeff}
    C_defs = [
        {0: 1, 3: 1, 4: -1, 6: -1}, # C00 from M1, M4, M5, M7
        {2: 1, 4: 1},               # C01 from M3, M5
        {1: 1, 3: 1},               # C10 from M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11 from M1, M2, M3, M6
    ]
    
    # Wait, the C_defs mapping above is "Which M contributes to which C block with what coeff".
    # I need to iterate over M, and for each M, add to the appropriate C blocks.
    
    for m_idx in range(7):
        x_coeffs, y_coeffs = M_defs[m_idx]
        
        for s in range(7):
            # Get the 2x2 Strassen term s
            # We need to create a global term
            # U, V, W will be filled
            
            # Iterate over local 2x2 indices a,b for U2, V2, W2
            for a in range(2):
                for b in range(2):
                    # U2[a, s] is coeff for X_ab
                    # X_ab = sum_{block} c_X[block] * A_block[local_a, local_b]
                    
                    # Determine which global A index corresponds to local (a,b) in which block
                    # This is tricky because X is a sum of blocks.
                    # We need to add contributions from each block to the global U.
                    
                    for block_id, c_x in x_coeffs.items():
                        if c_x == 0: continue
                        r_start, r_end, c_start, c_end = blocks[block_id]
                        # Local a,b in 2x2 block
                        # Global row = r_start + a
                        # Global col = c_start + b
                        g_row = r_start + a
                        g_col = c_start + b
                        g_u_idx = g_row * 5 + g_col
                        
                        coeff_u = c_x * U2[a, s]
                        if coeff_u != 0:
                            U[g_u_idx, col] += coeff_u

                    # Similarly for V
                    for block_id, c_y in y_coeffs.items():
                        if c_y == 0: continue
                        r_start, r_end, c_start, c_end = blocks[block_id]
                        g_row = r_start + b # Note: V index is for B, so B[j,k]. j is row, k is col.
                        # In 2x2 Strassen, V2[b, s] corresponds to B_b? 
                        # My U2/V2/W2 indices:
                        # U2 is for A. V2 is for B.
                        # A is 2x2. B is 2x2.
                        # V2 index b corresponds to row b of B?
                        # In my 2x2 Strassen definition:
                        # U2[0,0]=1 (A00), U2[1,0]=1 (A11).
                        # V2[0,0]=1 (B00), V2[1,0]=1 (B11).
                        # So V2 index is row of B.
                        # B_block is 2x2.
                        # Local row b, local col?
                        # Wait, V2 shape is (4,7).
                        # V2[0] is B00, V2[1] is B01, V2[2] is B10, V2[3] is B11.
                        # So V2 index is linear index of B element.
                        # My loop `for b in range(2)` above was for U2.
                        # I need to iterate over all 4 elements of B for V2.
                        
                        # Let's restart the inner loop logic for clarity.
                        # I will iterate over all 4 elements of X (U2) and all 4 elements of Y (V2).
                        pass

    # The above logic is getting messy. I will rewrite the inner loop properly.
    
    # Reset col to 0? No, I already used col? 
    # I haven't added any terms yet in the loop above because I passed.
    # I will clear U,V,W and restart the loop implementation.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0

    for m_idx in range(7):
        x_coeffs, y_coeffs = M_defs[m_idx]
        
        for s in range(7):
            # Iterate over 4 elements of X (U2)
            # U2 indices: 0:A00, 1:A01, 2:A10, 3:A11
            for u_idx in range(4):
                if U2[u_idx, s] == 0: continue
                # u_idx corresponds to (a,b) in 2x2
                a = u_idx // 2
                b = u_idx % 2
                # X_ab = sum_{block} c_x[block] * A_block[a,b]
                for block_id, c_x in x_coeffs.items():
                    if c_x == 0: continue
                    r_start, r_end, c_start, c_end = blocks[block_id]
                    g_row = r_start + a
                    g_col = c_start + b
                    g_u_idx = g_row * 5 + g_col
                    U[g_u_idx, col] += c_x * U2[u_idx, s]
                    
            # Iterate over 4 elements of Y (V2)
            for v_idx in range(4):
                if V2[v_idx, s] == 0: continue
                # v_idx corresponds to (b,k) in 2x2? 
                # V2 index is linear index of B.
                # B is 2x2.
                # b = v_idx // 2 (row), k = v_idx % 2 (col)
                b = v_idx // 2
                k = v_idx % 2
                # Y_bk = sum_{block} c_y[block] * B_block[b,k]
                for block_id, c_y in y_coeffs.items():
                    if c_y == 0: continue
                    r_start, r_end, c_start, c_end = blocks[block_id]
                    g_row = r_start + b
                    g_col = c_start + k
                    g_v_idx = g_row * 5 + g_col
                    V[g_v_idx, col] += c_y * V2[v_idx, s]
                    
            # Iterate over 4 elements of Z (W2)
            # W2 indices: 0:C00, 1:C01, 2:C10, 3:C11
            for w_idx in range(4):
                if W2[w_idx, s] == 0: continue
                # w_idx corresponds to (k,i) in 2x2 result
                k = w_idx // 2
                i = w_idx % 2
                # Z_ki contributes to C blocks
                # We need to find which C blocks this M contributes to
                # C_defs[m_idx] gives {c_block_id: coeff}
                # c_block_id 0: C00, 1: C01, 2: C10, 3: C11
                for c_block_id, c_z in C_defs[m_idx].items():
                    if c_z == 0: continue
                    r_start, r_end, c_start, c_end = blocks[c_block_id]
                    g_row = r_start + k # C row
                    g_col = c_start + i # C col
                    # W index is k*n + i. Here C is 5x5.
                    # W[g_col * 5 + g_row] ?
                    # Prompt: W[k*n+i, r].
                    # C is p x n. p=5, n=5.
                    # C row is k, C col is i.
                    # So W index is k*5 + i.
                    g_w_idx = g_row * 5 + g_col
                    W[g_w_idx, col] += c_z * W2[w_idx, s]
                    
            col += 1
            
    # Now add the other 61 terms
    # 2. A41 B14 (16)
    for i in range(4):
        for j in range(4):
            add_term(5*i + 4, 4*5 + j, 5*j + i, 1)
            
    # 3. A44 B41 (16)
    for i in range(4):
        for j in range(4):
            add_term(5*i + j, 5*j + 4, 5*4 + i, 1)
            
    # 4. A41 B11 (4)
    for i in range(4):
        add_term(5*i + 4, 4*5 + 4, 5*4 + i, 1)
        
    # 5. A14 B44 (16)
    for i in range(4):
        for j in range(4):
            add_term(5*4 + i, 5*i + j, 5*j + 4, 1)
            
    # 6. A11 B14 (4)
    for j in range(4):
        add_term(5*4 + 4, 4*5 + j, 5*j + 4, 1)
        
    # 7. A14 B41 (4)
    for i in range(4):
        add_term(5*4 + i, 5*i + 4, 5*4 + 4, 1)
        
    # 8. A11 B11 (1)
    add_term(5*4 + 4, 4*5 + 4, 5*4 + 4, 1)
    
    return {"U": U, "V": V, "W": W}
