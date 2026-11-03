import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Helper to expand 2x2 Strassen terms
    # u, v, w are lists of 4 coefficients
    # Indices: 00, 01, 10, 11
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

    def map_4x4_indices(block_row, block_col, local_idx):
        # block_row, block_col in {0,1}
        # local_idx in {0,1,2,3} corresponding to 00,01,10,11
        # Global index = (block_row*2 + local_row)*4 + (block_col*2 + local_col)
        lr = local_idx // 2
        lc = local_idx % 2
        gr = block_row * 2 + lr
        gc = block_col * 2 + lc
        return gr * 4 + gc

    def expand_4x4_strassen():
        terms = []
        # 7 block terms
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        block_defs = [
            # (A_blocks, B_blocks, C_contribution_signs)
            # A_blocks: list of (block_idx, coeff)
            # B_blocks: list of (block_idx, coeff)
            # C_blocks: list of (block_idx, coeff)
            
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2
            ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
            # M3
            ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
            # M4
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            # M5
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            # M6
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            # M7
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
        ]
        
        for a_blocks, b_blocks, c_blocks in block_defs:
            # For each of the 7 scalar terms in 2x2 Strassen
            for u2, v2, w2 in strassen_2x2:
                u4 = [0] * 16
                v4 = [0] * 16
                w4 = [0] * 16
                
                for block_idx, coeff_a in a_blocks:
                    br = block_idx // 2
                    bc = block_idx % 2
                    for li in range(4):
                        gi = map_4x4_indices(br, bc, li)
                        u4[gi] += coeff_a * u2[li]
                        
                for block_idx, coeff_b in b_blocks:
                    br = block_idx // 2
                    bc = block_idx % 2
                    for li in range(4):
                        gi = map_4x4_indices(br, bc, li)
                        v4[gi] += coeff_b * v2[li]
                        
                for block_idx, coeff_c in c_blocks:
                    br = block_idx // 2
                    bc = block_idx % 2
                    for li in range(4):
                        gi = map_4x4_indices(br, bc, li)
                        w4[gi] += coeff_c * w2[li]
                        
                terms.append((u4, v4, w4))
        return terms

    def expand_4x4x2():
        # A is 4x4, B is 4x2, C is 4x2
        # B = [B0; B1] where B0, B1 are 2x2
        # C = [C0; C1] where C0, C1 are 2x2
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Each term is sum of two 2x2 products.
        # We use 7 terms for each product.
        
        terms = []
        
        # C0 = A00 B0 + A01 B1
        # Product 1: A00 B0
        # Product 2: A01 B1
        
        # C1 = A10 B0 + A11 B1
        # Product 3: A10 B0
        # Product 4: A11 B1
        
        products = [
            # (A_block, B_block, C_block)
            (0, 0, 0), # A00 B0 -> C0
            (1, 1, 0), # A01 B1 -> C0
            (2, 0, 1), # A10 B0 -> C1
            (3, 1, 1)  # A11 B1 -> C1
        ]
        
        for a_block, b_block, c_block in products:
            for u2, v2, w2 in strassen_2x2:
                u4 = [0] * 16
                v8 = [0] * 8
                w8 = [0] * 8
                
                br_a = a_block // 2
                bc_a = a_block % 2
                for li in range(4):
                    gi = map_4x4_indices(br_a, bc_a, li)
                    u4[gi] = u2[li]
                    
                # B is 4x2.
                # B_block 0 is top 2x2 (rows 0,1 of B? No, B is 4x2.
                # B = [B0; B1]. B0 is rows 0,1? No.
                # B is 4x2.
                # B0 is 2x2. Which rows?
                # In our previous derivation:
                # A = [A00 A01; A10 A11]
                # B = [B0; B1]
                # B0 is 2x2.
                # B0 corresponds to B[0:2, 0:2]?
                # Let's check dimensions.
                # A is 4x4. B is 4x2.
                # C is 4x2.
                # C0 = A00 B0 + A01 B1.
                # A00 is 2x2. B0 must be 2x2.
                # A00 is rows 0,1 of A.
                # So B0 must be rows 0,1 of B.
                # B1 must be rows 2,3 of B.
                
                # So B0 is B[0:2, 0:2].
                # B1 is B[2:4, 0:2].
                
                # Global B index: row * 2 + col.
                # B0 rows 0,1. Cols 0,1.
                # B1 rows 2,3. Cols 0,1.
                
                br_b = b_block # 0 or 1
                # b_block 0 -> rows 0,1
                # b_block 1 -> rows 2,3
                start_row = br_b * 2
                for li in range(4):
                    lr = li // 2
                    lc = li % 2
                    gr = start_row + lr
                    gc = lc
                    gi = gr * 2 + gc
                    v8[gi] = v2[li]
                    
                # C is 4x2.
                # C0 is rows 0,1.
                # C1 is rows 2,3.
                br_c = c_block # 0 or 1
                start_row_c = br_c * 2
                for li in range(4):
                    lr = li // 2
                    lc = li % 2
                    gr = start_row_c + lr
                    gc = lc
                    gi = gr * 2 + gc
                    w8[gi] = w2[li]
                    
                terms.append((u4, v8, w8))
                
        return terms

    terms_4x4 = expand_4x4_strassen()
    terms_4x4x2 = expand_4x4x2()
    
    # Combine
    # U: 16 x rank
    # V: 24 x rank
    # W: 24 x rank
    
    rank = len(terms_4x4) + len(terms_4x4x2)
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for u, v, w in terms_4x4:
        for i in range(16):
            U[i][col] = u[i]
        for i in range(16):
            V[i][col] = v[i]
        for i in range(16):
            W[i][col] = w[i]
        col += 1
        
    for u, v, w in terms_4x4x2:
        for i in range(16):
            U[i][col] = u[i]
        # v is length 8. Global V is length 24.
        # B2 is last 2 cols of B.
        # Global B index: j * 6 + k.
        # B2 cols are k=4,5.
        # Local v index: j * 2 + k_local.
        # k_local 0 -> k=4.
        # k_local 1 -> k=5.
        for i in range(8):
            j = i // 2
            k_local = i % 2
            k_global = 4 + k_local
            gi = j * 6 + k_global
            V[gi][col] = v[i]
            
        # w is length 8. Global W is length 24.
        # C2 is last 2 cols of C.
        # Global W index: k * 4 + i.
        # C2 cols are k=4,5.
        # Local w index: k_local * 4 + i.
        for i in range(8):
            k_local = i // 4
            i_global = i % 4
            k_global = 4 + k_local
            gi = k_global * 4 + i_global
            W[gi][col] = w[i]
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
