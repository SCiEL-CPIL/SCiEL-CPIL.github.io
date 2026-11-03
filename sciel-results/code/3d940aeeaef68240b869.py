import numpy as np

DIMENSIONS = [4, 4, 6]

def get_strassen_terms_2x2():
    """
    Returns list of (U, V, W) for 2x2x2 Strassen decomposition.
    U, V, W are lists of length 4.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # W coefficients for C00, C01, C10, C11
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    w_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 0, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [-1, 1, 0, 0]  # M7
    ]
    
    u_coeffs = [
        [1, 0, 1, 0], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [0, 0, 1, -1],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    v_coeffs = [
        [1, 1, 0, 0], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [0, 1, -1, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 0, 1, 0], # M6: B00+B10
        [0, 1, 0, 1]  # M7: B01+B11
    ]
    
    for i in range(7):
        terms.append((u_coeffs[i], v_coeffs[i], w_coeffs[i]))
    return terms

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: 16 x R
    # V: 24 x R
    # W: 24 x R
    
    # We will build the rank 1 terms as lists of (u, v, w)
    # u is length 16, v is length 24, w is length 24
    
    terms = []
    
    # Part 1: 4x4x4 multiplication (Rank 49)
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols)
    # We use recursive Strassen.
    
    def expand_strassen_4x4():
        sub_terms = get_strassen_terms_2x2()
        # Each sub-term is for 2x2x2.
        # We need to apply this to the 4x4 blocks.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # The 7 products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B21)
        # M7 = (A12-A22)(B12+B22)
        
        # Each Mq is a 2x2 matrix product.
        # We expand each Mq using the 2x2 Strassen terms.
        
        # Mapping of block indices to global indices
        # A blocks:
        # A11: rows 0-1, cols 0-1
        # A12: rows 0-1, cols 2-3
        # A21: rows 2-3, cols 0-1
        # A22: rows 2-3, cols 2-3
        
        def get_block_indices(block_id):
            # block_id: 0=A11, 1=A12, 2=A21, 3=A22
            r, c = divmod(block_id, 2)
            return (r*2, r*2+1, c*2, c*2+1)
            
        # For each of the 7 meta-products, we have 7 sub-products.
        # Total 49 terms.
        
        # Meta product definitions (U_block, V_block, W_block_contrib)
        # U_block: which A blocks are involved and with what sign
        # V_block: which B blocks are involved and with what sign
        # W_block: which C blocks are involved and with what sign (from 2x2 Strassen W coeffs)
        
        # M1: A(11+22), B(11+22), C(11+22? No, C is result)
        # The W coefficients from get_strassen_terms_2x2 give the contribution to the 2x2 output blocks.
        # So if M1 is a 2x2 matrix, its entries contribute to C11, C12, C21, C22 based on the W coeffs.
        
        # Let's iterate through the 7 meta-products.
        
        meta_u_blocks = [
            {0: 1, 3: 1}, # M1: A11 + A22
            {2: 1, 3: 1}, # M2: A21 + A22
            {0: 1},       # M3: A11
            {3: 1},       # M4: A22
            {0: 1, 1: 1}, # M5: A11 + A12
            {2: 1, 0: -1},# M6: A21 - A11
            {1: 1, 3: -1} # M7: A12 - A22
        ]
        
        meta_v_blocks = [
            {0: 1, 3: 1}, # M1: B11 + B22
            {0: 1},       # M2: B11
            {2: 1, 3: -1},# M3: B21 - B22
            {1: 1, 0: -1},# M4: B12 - B11
            {3: 1},       # M5: B22
            {0: 1, 2: 1}, # M6: B11 + B21
            {1: 1, 3: 1}  # M7: B12 + B22
        ]
        
        # W coefficients for the 2x2 output blocks (C11, C12, C21, C22)
        # From get_strassen_terms_2x2, w_coeffs[i] gives coeffs for C00, C01, C10, C11 of the 2x2 result.
        # These correspond to C11, C12, C21, C22 in the 4x4 case.
        
        w_2x2 = get_strassen_terms_2x2()
        
        for meta_idx in range(7):
            # Get the 7 sub-terms for this meta-product
            for sub_idx in range(7):
                u_sub, v_sub, w_sub = w_2x2[sub_idx]
                
                # Construct global U, V, W vectors
                u_global = [0] * 16
                v_global = [0] * 24
                w_global = [0] * 24
                
                # Fill U
                for block_id, coeff in meta_u_blocks[meta_idx].items():
                    r1, r2, c1, c2 = get_block_indices(block_id)
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            # Map sub-index (r-r1, c-c1) to u_sub
                            sub_r = r - r1
                            sub_c = c - c1
                            sub_idx_u = sub_r * 2 + sub_c
                            u_global[r * 4 + c] = coeff * u_sub[sub_idx_u]
                            
                # Fill V
                # B blocks are same layout as A
                for block_id, coeff in meta_v_blocks[meta_idx].items():
                    r1, r2, c1, c2 = get_block_indices(block_id)
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            sub_r = r - r1
                            sub_c = c - c1
                            sub_idx_v = sub_r * 2 + sub_c
                            # B is 4x4 (first 4 cols)
                            v_global[r * 4 + c] = coeff * v_sub[sub_idx_v]
                            
                # Fill W
                # w_sub gives coeffs for C11, C12, C21, C22
                # w_sub[0] -> C11 (rows 0-1, cols 0-1)
                # w_sub[1] -> C12 (rows 0-1, cols 2-3)
                # w_sub[2] -> C21 (rows 2-3, cols 0-1)
                # w_sub[3] -> C22 (rows 2-3, cols 2-3)
                
                for block_id in range(4):
                    coeff_w = w_sub[block_id]
                    if coeff_w == 0: continue
                    r1, r2, c1, c2 = get_block_indices(block_id)
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            sub_r = r - r1
                            sub_c = c - c1
                            # w_global index: k*n + i? 
                            # Task says W[k*n+i, r].
                            # C is 4x4. i is row, k is col.
                            # So index is c * 4 + r.
                            w_global[c * 4 + r] = coeff_w
                            
                terms.append((u_global, v_global, w_global))
                
    expand_strassen_4x4()
    
    # Part 2: 4x4x2 multiplication (Rank 28)
    # A is 4x4, B is 4x2 (last 2 cols), C is 4x2 (last 2 cols)
    # B_right = [[B_top], [B_bot]] where B_top, B_bot are 2x2
    # C_right = [[C_top], [C_bot]] where C_top, C_bot are 2x2
    
    # Products:
    # P1 = A11 B_top -> C_top_left? No.
    # C_top = A11 B_top + A12 B_bot
    # C_bot = A21 B_top + A22 B_bot
    
    # So we have 4 independent 2x2x2 multiplications.
    # 1. A11 * B_top -> contributes to C_top
    # 2. A12 * B_bot -> contributes to C_top
    # 3. A21 * B_top -> contributes to C_bot
    # 4. A22 * B_bot -> contributes to C_bot
    
    # Each is a 2x2x2 Strassen.
    
    w_2x2 = get_strassen_terms_2x2()
    
    def get_block_indices_4x4(block_id):
        # 0=A11, 1=A12, 2=A21, 3=A22
        r, c = divmod(block_id, 2)
        return (r*2, r*2+1, c*2, c*2+1)
        
    # B_top: rows 0-1, cols 4-5
    # B_bot: rows 2-3, cols 4-5
    # C_top: rows 0-1, cols 4-5
    # C_bot: rows 2-3, cols 4-5
    
    products = [
        (0, 'top'), # A11 * B_top
        (1, 'bot'), # A12 * B_bot
        (2, 'top'), # A21 * B_top
        (3, 'bot')  # A22 * B_bot
    ]
    
    for a_block, b_part in products:
        for sub_idx in range(7):
            u_sub, v_sub, w_sub = w_2x2[sub_idx]
            
            u_global = [0] * 16
            v_global = [0] * 24
            w_global = [0] * 24
            
            # Fill U
            r1, r2, c1, c2 = get_block_indices_4x4(a_block)
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    sub_r = r - r1
                    sub_c = c - c1
                    sub_idx_u = sub_r * 2 + sub_c
                    u_global[r * 4 + c] = u_sub[sub_idx_u]
                    
            # Fill V
            if b_part == 'top':
                v_r1, v_r2 = 0, 1
            else:
                v_r1, v_r2 = 2, 3
            v_c1, v_c2 = 4, 5
            
            for r in range(v_r1, v_r2+1):
                for c in range(v_c1, v_c2+1):
                    sub_r = r - v_r1
                    sub_c = c - v_c1
                    sub_idx_v = sub_r * 2 + sub_c
                    v_global[r * 4 + c] = v_sub[sub_idx_v]
                    
            # Fill W
            # The result of A_block * B_part is a 2x2 matrix.
            # It contributes to C_top or C_bot.
            # w_sub gives coeffs for the 2x2 result.
            # The result is placed in C.
            
            if b_part == 'top':
                # Contributes to C_top (rows 0-1, cols 4-5)
                w_r1, w_r2 = 0, 1
            else:
                w_r1, w_r2 = 2, 3
            w_c1, w_c2 = 4, 5
            
            # w_sub[0] -> C[r1, c1]
            # w_sub[1] -> C[r1, c2]
            # w_sub[2] -> C[r2, c1]
            # w_sub[3] -> C[r2, c2]
            
            # Map w_sub indices to global W indices (k*n + i)
            # w_global[c * 4 + r]
            
            for sub_r in range(2):
                for sub_c in range(2):
                    coeff = w_sub[sub_r * 2 + sub_c]
                    if coeff == 0: continue
                    r = w_r1 + sub_r
                    c = w_c1 + sub_c
                    w_global[c * 4 + r] = coeff
                    
            terms.append((u_global, v_global, w_global))
            
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    for r_idx, (u, v, w) in enumerate(terms):
        for i in range(16):
            U[i][r_idx] = u[i]
        for i in range(24):
            V[i][r_idx] = v[i]
            W[i][r_idx] = w[i]
            
    return {"U": U, "V": V, "W": W}
