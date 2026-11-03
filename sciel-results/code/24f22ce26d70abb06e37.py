import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank-56 decomposition.
    # This is based on two independent rank-28 decompositions for 2x4x4 multiplications.
    # Each 2x4x4 multiplication is decomposed into 4 Strassen 2x2x2 multiplications.
    # Rank per 2x4x4 = 4 * 7 = 28.
    # Total rank = 28 + 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = A11 * (B12 - B22)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 + A22) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A21) * (B11 + B22)
    # M6 = (A12 - A21) * (B21 + B22)
    # M7 = (A11 - A22) * (B11 + B12)
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M7
    # C22 = M1 - M3 + M5 + M4
    
    # We define the 7 linear combinations for A and B
    # A indices in 2x2 block: 0:11, 1:12, 2:21, 3:22
    # B indices in 2x2 block: 0:11, 1:12, 2:21, 3:22
    
    a_combos = [
        [1, 0, 0, 0], # A11
        [1, 1, 0, 0], # A11+A12
        [0, 0, 1, 1], # A21+A22
        [0, 0, 0, 1], # A22
        [1, 0, 1, 0], # A11+A21
        [0, 1, -1, 0],# A12-A21
        [1, 0, 0, -1] # A11-A22
    ]
    
    b_combos = [
        [0, 1, 0, -1], # B12-B22
        [0, 0, 0, 1],  # B22
        [1, 0, 0, 0],  # B11
        [0, 0, 1, -1], # B21-B11
        [1, 0, 0, 1],  # B11+B22
        [0, 0, 1, 1],  # B21+B22
        [1, 1, 0, 0]   # B11+B12
    ]
    
    c_combos = [
        [1, 0, 0, 0], # M5+M4-M2+M6 -> C11
        [0, 1, 0, 0], # M1+M5 -> C12
        [0, 0, 1, 0], # M3+M7 -> C21
        [0, 0, 0, 1]  # M1-M3+M5+M4 -> C22
    ]
    
    # Wait, the c_combos above are for the 4 outputs of a 2x2x2 Strassen.
    # We need to map these to the global indices.
    
    def add_strassen_block(offset_u, offset_v, offset_w, col_start):
        # offset_u, offset_v, offset_w are starting indices for the 2x2 blocks in A, B, C
        # A is 2x4. We split into two 2x2 blocks horizontally: A_left (cols 0-1), A_right (cols 2-3)
        # B is 4x4. We split into four 2x2 blocks: B_TL (rows 0-1, cols 0-1), B_TR (rows 0-1, cols 2-3), B_BL (rows 2-3, cols 0-1), B_BR (rows 2-3, cols 2-3)
        # C is 2x4. We split into two 2x2 blocks horizontally: C_left (cols 0-1), C_right (cols 2-3)
        
        # The 4 products are:
        # P1: A_left * B_TL -> C_left_top
        # P2: A_right * B_BL -> C_left_bottom
        # P3: A_left * B_TR -> C_right_top
        # P4: A_right * B_BR -> C_right_bottom
        
        # Note: A is 2x4. A_left is rows 0-1, cols 0-1. A_right is rows 0-1, cols 2-3.
        # B is 4x4. B_TL is rows 0-1, cols 0-1. B_BL is rows 2-3, cols 0-1.
        # B_TR is rows 0-1, cols 2-3. B_BR is rows 2-3, cols 2-3.
        # C is 2x4. C_left is rows 0-1, cols 0-1. C_right is rows 0-1, cols 2-3.
        
        # For each of the 4 products, we use 7 Strassen terms.
        for prod_idx in range(4):
            # Determine which A block, B block, C block
            if prod_idx == 0:
                a_block = 0 # A_left
                b_block = 0 # B_TL
                c_block = 0 # C_left
            elif prod_idx == 1:
                a_block = 1 # A_right
                b_block = 2 # B_BL
                c_block = 0 # C_left
            elif prod_idx == 2:
                a_block = 0 # A_left
                b_block = 1 # B_TR
                c_block = 1 # C_right
            else:
                a_block = 1 # A_right
                b_block = 3 # B_BR
                c_block = 1 # C_right
            
            for s in range(7):
                col = col_start + prod_idx * 7 + s
                
                # Fill U
                # A block is 2x2. Global A index: row i, col j.
                # a_block 0: cols 0,1. a_block 1: cols 2,3.
                # Local A index in 2x2: 0:00, 1:01, 2:10, 3:11.
                for local_a in range(4):
                    val = a_combos[s][local_a]
                    if val != 0:
                        # Map local_a to global A index
                        # local_a 0 -> r0 c0
                        # local_a 1 -> r0 c1
                        # local_a 2 -> r1 c0
                        # local_a 3 -> r1 c1
                        r = local_a // 2
                        c = local_a % 2
                        global_c = a_block * 2 + c
                        global_idx = r * m + global_c
                        U[global_idx][col] += val
                        
                # Fill V
                # B block is 2x2. Global B index: row j, col k.
                # b_block 0 (TL): rows 0,1 cols 0,1
                # b_block 1 (TR): rows 0,1 cols 2,3
                # b_block 2 (BL): rows 2,3 cols 0,1
                # b_block 3 (BR): rows 2,3 cols 2,3
                for local_b in range(4):
                    val = b_combos[s][local_b]
                    if val != 0:
                        # Map local_b to global B index
                        r = local_b // 2
                        c = local_b % 2
                        # Determine global row and col for B
                        if b_block == 0:
                            gr = r
                            gc = c
                        elif b_block == 1:
                            gr = r
                            gc = 2 + c
                        elif b_block == 2:
                            gr = 2 + r
                            gc = c
                        else:
                            gr = 2 + r
                            gc = 2 + c
                        global_idx = gr * p + gc
                        V[global_idx][col] += val
                        
                # Fill W
                # C block is 2x2. Global C index: row k, col i.
                # c_block 0 (Left): cols 0,1
                # c_block 1 (Right): cols 2,3
                # The Strassen output C is 2x2.
                # We need to add the 7 products to the C blocks.
                # The combination of M1..M7 to get C11..C22 is fixed.
                # We need to distribute the 7 terms to the 4 C positions.
                # c_combos[s][local_c] is the coefficient for local_c in the s-th term?
                # No, c_combos defined earlier was for the 4 outputs.
                # Let's use the standard Strassen reconstruction:
                # C11 = M5 + M4 - M2 + M6
                # C12 = M1 + M5
                # C21 = M3 + M7
                # C22 = M1 - M3 + M5 + M4
                
                # So for term s (M_{s+1}), we add to C positions with specific coeffs.
                # s=0 (M1): C12 (+1), C22 (+1)
                # s=1 (M2): C11 (-1)
                # s=2 (M3): C21 (+1), C22 (-1)
                # s=3 (M4): C11 (+1), C22 (+1)
                # s=4 (M5): C11 (+1), C12 (+1), C22 (+1)
                # s=5 (M6): C11 (+1)
                # s=6 (M7): C21 (+1)
                
                coeffs = [
                    [0, 1, 0, 0], # M1 -> C12, C22? No, C12 is local 1, C22 is local 3.
                    [0, 0, 0, 0], # M2 -> C11 (-1). Local 0.
                    [0, 0, 0, 0], # M3 -> C21 (+1), C22 (-1). Local 2, 3.
                    [0, 0, 0, 0], # M4 -> C11 (+1), C22 (+1). Local 0, 3.
                    [0, 0, 0, 0], # M5 -> C11 (+1), C12 (+1), C22 (+1). Local 0, 1, 3.
                    [0, 0, 0, 0], # M6 -> C11 (+1). Local 0.
                    [0, 0, 0, 0]  # M7 -> C21 (+1). Local 2.
                ]
                # Correct coeffs:
                # M1: C12(1), C22(1) -> [0, 1, 0, 1]
                # M2: C11(-1) -> [-1, 0, 0, 0]
                # M3: C21(1), C22(-1) -> [0, 0, 1, -1]
                # M4: C11(1), C22(1) -> [1, 0, 0, 1]
                # M5: C11(1), C12(1), C22(1) -> [1, 1, 0, 1]
                # M6: C11(1) -> [1, 0, 0, 0]
                # M7: C21(1) -> [0, 0, 1, 0]
                
                c_coeffs = [
                    [0, 1, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 0, 1, -1],
                    [1, 0, 0, 1],
                    [1, 1, 0, 1],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0]
                ]
                
                for local_c in range(4):
                    val = c_coeffs[s][local_c]
                    if val != 0:
                        # Map local_c to global C index
                        # C is 2x4. W index is k*n+i.
                        # local_c 0: r0 c0
                        # local_c 1: r0 c1
                        # local_c 2: r1 c0
                        # local_c 3: r1 c1
                        r = local_c // 2
                        c = local_c % 2
                        global_c = c_block * 2 + c
                        # W index: k*n + i. Here r is k (row of C), c is i (col of C)?
                        # Task says W[k*n+i, r]. C is p x n.
                        # C row k, col i.
                        # In our C block, r is row index (0,1), c is col index (0,1).
                        # So k = r, i = c.
                        global_idx = r * n + global_c
                        W[global_idx][col] += val

    # First 4 columns of B (cols 0-3)
    add_strassen_block(0, 0, 0, 0)
    # Next 4 columns of B (cols 4-7)
    # We need to shift the B indices.
    # The function above assumes B is 4x4.
    # For the second half, B is rows 0-3, cols 4-7.
    # We can just call it again with an offset for V and W?
    # No, the function hardcodes the B block positions.
    # We need a more flexible function or just duplicate the logic with offsets.
    
    # Let's refactor to allow offsets.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block_v2(a_col_offset, b_col_offset, c_col_offset, col_start):
        # a_col_offset: 0 or 2 (for A_left or A_right) - but we use both in one block?
        # No, the 4 products use A_left and A_right.
        # This function adds one full 2x4x4 decomposition.
        # a_col_offset is not used for A because A is always the full 2x4.
        # b_col_offset: 0 or 4.
        # c_col_offset: 0 or 4.
        
        for prod_idx in range(4):
            if prod_idx == 0:
                a_block = 0 # A_left
                b_local_block = 0 # B_TL relative to b_col_offset
                c_block = 0 # C_left relative to c_col_offset
            elif prod_idx == 1:
                a_block = 1 # A_right
                b_local_block = 2 # B_BL
                c_block = 0 # C_left
            elif prod_idx == 2:
                a_block = 0 # A_left
                b_local_block = 1 # B_TR
                c_block = 1 # C_right
            else:
                a_block = 1 # A_right
                b_local_block = 3 # B_BR
                c_block = 1 # C_right
            
            for s in range(7):
                col = col_start + prod_idx * 7 + s
                
                # U
                for local_a in range(4):
                    val = a_combos[s][local_a]
                    if val != 0:
                        r = local_a // 2
                        c = local_a % 2
                        global_c = a_block * 2 + c
                        global_idx = r * m + global_c
                        U[global_idx][col] += val
                        
                # V
                for local_b in range(4):
                    val = b_combos[s][local_b]
                    if val != 0:
                        r = local_b // 2
                        c = local_b % 2
                        # b_local_block determines row/col shift within the 4x4 B slice
                        if b_local_block == 0:
                            gr = r
                            gc = c
                        elif b_local_block == 1:
                            gr = r
                            gc = 2 + c
                        elif b_local_block == 2:
                            gr = 2 + r
                            gc = c
                        else:
                            gr = 2 + r
                            gc = 2 + c
                        
                        # Add global offset
                        global_row = gr
                        global_col = b_col_offset + gc
                        global_idx = global_row * p + global_col
                        V[global_idx][col] += val
                        
                # W
                c_coeffs = [
                    [0, 1, 0, 1],
                    [-1, 0, 0, 0],
                    [0, 0, 1, -1],
                    [1, 0, 0, 1],
                    [1, 1, 0, 1],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0]
                ]
                for local_c in range(4):
                    val = c_coeffs[s][local_c]
                    if val != 0:
                        r = local_c // 2
                        c = local_c % 2
                        global_c = c_block * 2 + c
                        global_idx = r * n + (c_col_offset + global_c)
                        W[global_idx][col] += val

    add_strassen_block_v2(0, 0, 0, 0)
    add_strassen_block_v2(0, 4, 4, 28)

    return {"U": U, "V": V, "W": W}

