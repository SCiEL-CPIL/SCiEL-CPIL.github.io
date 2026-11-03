import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Total rank calculation:
    # 4x4x4 Strassen: 49
    # 4x4x2 Strassen: 28 (for cols 7-8)
    # 4x4x3: 16 (col 4) + 28 (cols 5-6) = 44
    # Total: 49 + 28 + 44 = 121
    
    rank_total = 121
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    current_col = 0
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_col
        if current_col >= rank_total:
            raise ValueError("Rank overflow")
        for idx, val in zip(u_indices, u_vals):
            U[idx, current_col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, current_col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, current_col] += val
        current_col += 1

    # Strassen 2x2x2 terms
    # Indices: 0:11, 1:12, 2:21, 3:22
    strassen_222 = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # m1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),       # m2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),       # m3
        ([3], [1], [1, 2], [-1, 1], [0, 2], [1, 1]),       # m4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),       # m5
        ([0, 2], [-1, 1], [0, 1], [1, 1], [3], [1]),       # m6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])        # m7
    ]

    def add_strassen_222(u_base, v_base, w_base):
        for term in strassen_222:
            u_idx = [u_base + i for i in term[0]]
            u_val = term[1]
            v_idx = [v_base + i for i in term[2]]
            v_val = term[3]
            w_idx = [w_base + i for i in term[4]]
            w_val = term[5]
            add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val)

    # Strassen 4x4x4
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]]
    # A11 is 2x2 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B11 is 2x2 at (0,0), B12 at (0,2), B21 at (2,0), B22 at (2,2)
    # C11 is 2x2 at (0,0), C12 at (0,2), C21 at (2,0), C22 at (2,2)
    
    # Strassen 4x4x4 uses 7 multiplications of 2x2x2? No.
    # Strassen 4x4x4 is usually done by 7 calls to 2x2x2? No, that's 7*7=49 rank.
    # Yes, 49 is the rank.
    
    # The 7 products for 4x4x4 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these 2x2x2 multiplications to the global 4x4x4 indices.
    # Each M is a 2x2x2 multiplication.
    # M1: A part is A11+A22. This is not a single 2x2 block, it's a sum.
    # So we can't just call add_strassen_222 on a block.
    # We must decompose the sums.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This requires 4 rank-1 terms? No, Strassen works on the blocks.
    # The standard Strassen 4x4x4 implementation uses 7 calls to the 2x2x2 Strassen algorithm?
    # No, the 2x2x2 Strassen algorithm computes the product of two 2x2 matrices.
    # If we apply it to 4x4 matrices, we treat them as 2x2 blocks of 2x2 matrices.
    # The "multiplication" step in Strassen 4x4x4 is the multiplication of 2x2 matrices.
    # So yes, we call the 2x2x2 Strassen decomposition 7 times.
    
    # However, the inputs to these 2x2x2 multiplications are sums/differences of blocks.
    # So the U and V vectors will have non-zero entries in multiple blocks.
    
    # Let's define the 7 calls for 4x4x4 Strassen.
    # Block indices: 0:11, 1:12, 2:21, 3:22 (for 2x2 blocks)
    # Global indices:
    # Block 0 (11): rows 0-1, cols 0-1
    # Block 1 (12): rows 0-1, cols 2-3
    # Block 2 (21): rows 2-3, cols 0-1
    # Block 3 (22): rows 2-3, cols 2-3
    
    # M1: (A11+A22)(B11+B22)
    # A side: A11 (block 0) + A22 (block 3)
    # B side: B11 (block 0) + B22 (block 3)
    # C side: C11 (block 0) + C22 (block 3)
    # We can't just use add_strassen_222 because it assumes single blocks.
    # We need a more flexible add function.
    
    # Let's rewrite add_term to accept full vectors.
    # Or, we can construct the U, V, W vectors for each of the 7 terms explicitly.
    
    # Term 1: M1
    # U: 1s in A11 and A22.
    # V: 1s in B11 and B22.
    # W: 1s in C11 and C22.
    # This is a single rank-1 term in the tensor decomposition of the 4x4x4 tensor?
    # No, M1 is the result of a 2x2x2 multiplication.
    # The 2x2x2 multiplication itself has rank 7.
    # So M1 contributes 7 rank-1 terms to the 4x4x4 decomposition.
    # The U, V, W vectors for these 7 terms are the Strassen 2x2x2 vectors,
    # but mapped to the global indices.
    
    # For M1, the "A" matrix is A11+A22.
    # The "B" matrix is B11+B22.
    # The "C" matrix is C11+C22.
    
    # The 7 terms of Strassen 2x2x2 for M1:
    # t1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # This corresponds to:
    # U: 1 in A11(11) and A22(22) -> Global: A[0,0], A[1,1], A[2,2], A[3,3]
    # V: 1 in B11(11) and B22(22) -> Global: B[0,0], B[1,1], B[2,2], B[3,3]
    # W: 1 in C11(11) and C22(22) -> Global: C[0,0], C[1,1], C[2,2], C[3,3]
    
    # This is getting complicated to map manually.
    # Let's use a helper that takes block offsets and signs.
    
    def add_strassen_444(u_blocks, v_blocks, w_blocks):
        # u_blocks, v_blocks, w_blocks are lists of (block_id, sign)
        # block_id: 0,1,2,3
        # This function adds the 7 terms of Strassen 2x2x2 for this specific product.
        
        # We need to construct the full U, V, W vectors for each of the 7 sub-terms.
        # This is complex.
        
        # Alternative: Just use the schoolbook 4x4x4? Rank 64.
        # 64 + 28 + 44 = 136. Still better than 144.
        # But 49 is better.
        
        # Let's try to implement the mapping.
        
        # For a given product (A_blocks * B_blocks -> C_blocks),
        # we iterate through the 7 Strassen 2x2x2 terms.
        # For each term, we have local U, V, W vectors of size 4.
        # We expand these to global vectors of size 16.
        
        for term in strassen_222:
            local_u_idx = term[0]
            local_u_val = term[1]
            local_v_idx = term[2]
            local_v_val = term[3]
            local_w_idx = term[4]
            local_w_val = term[5]
            
            global_u_idx = []
            global_u_val = []
            global_v_idx = []
            global_v_val = []
            global_w_idx = []
            global_w_val = []
            
            # Map local indices to global
            # local 0: block 0 (11), local 1: block 0 (12), local 2: block 0 (21), local 3: block 0 (22)
            # Wait, local indices are 0,1,2,3 corresponding to the 4 elements of the 2x2 matrix.
            # The block_id tells us which 2x2 block of the 4x4 matrix we are in.
            
            # For each local index, find the corresponding global index.
            # local 0 -> row 0, col 0 of the block
            # local 1 -> row 0, col 1 of the block
            # local 2 -> row 1, col 0 of the block
            # local 3 -> row 1, col 1 of the block
            
            # Block 0: rows 0-1, cols 0-1
            # Block 1: rows 0-1, cols 2-3
            # Block 2: rows 2-3, cols 0-1
            # Block 3: rows 2-3, cols 2-3
            
            def get_global_index(block_id, local_idx):
                row = (block_id // 2) * 2 + (local_idx // 2)
                col = (block_id % 2) * 2 + (local_idx % 2)
                return row * 4 + col
            
            for l_idx, val in zip(local_u_idx, local_u_val):
                for b_id, sign in u_blocks:
                    g_idx = get_global_index(b_id, l_idx)
                    global_u_idx.append(g_idx)
                    global_u_val.append(val * sign)
            
            for l_idx, val in zip(local_v_idx, local_v_val):
                for b_id, sign in v_blocks:
                    g_idx = get_global_index(b_id, l_idx)
                    global_v_idx.append(g_idx)
                    global_v_val.append(val * sign)
            
            for l_idx, val in zip(local_w_idx, local_w_val):
                for b_id, sign in w_blocks:
                    g_idx = get_global_index(b_id, l_idx)
                    global_w_idx.append(g_idx)
                    global_w_val.append(val * sign)
            
            add_term(global_u_idx, global_u_val, global_v_idx, global_v_val, global_w_idx, global_w_val)
            
    # Now define the 7 products for 4x4x4 Strassen
    # M1: (A11+A22)(B11+B22) -> C11+C22
    add_strassen_444([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
    # M2: (A21+A22)B11 -> C21
    add_strassen_444([(2,1), (3,1)], [(0,1)], [(2,1)])
    # M3: A11(B12-B22) -> C12
    add_strassen_444([(0,1)], [(1,1), (3,-1)], [(1,1)])
    # M4: A22(B21-B11) -> C11+C21
    add_strassen_444([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)])
    # M5: (A11+A12)B22 -> C11+C12
    add_strassen_444([(0,1), (1,1)], [(3,1)], [(0,1), (1,1)])
    # M6: (A21-A11)(B11+B12) -> C22
    add_strassen_444([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
    # M7: (A12-A22)(B21+B22) -> C11
    add_strassen_444([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    
    # 4x4x2 Strassen (for cols 7-8 of B, which are cols 3-4 of the 4x4x2 sub-problem)
    # B is 4x2. A is 4x4.
    # Split A into 2x2 blocks. B into 2x2 blocks (top and bottom).
    # B1 (2x2) at rows 0-1, B2 (2x2) at rows 2-3.
    # C1 (2x2) at rows 0-1, C2 (2x2) at rows 2-3.
    
    # C1 = A11B1 + A12B2
    # C2 = A21B1 + A22B2
    
    # Each product is 2x2x2.
    # A11B1: A block 0, B block 0 (rows 0-1, cols 0-1 of B), C block 0
    # A12B2: A block 1, B block 1 (rows 2-3, cols 0-1 of B), C block 0
    # A21B1: A block 2, B block 0, C block 1
    # A22B2: A block 3, B block 1, C block 1
    
    # We need to map B indices.
    # B is 4x2. Global V indices are 0..7.
    # B1 is rows 0-1, cols 0-1. V indices: 0,1, 4,5.
    # B2 is rows 2-3, cols 0-1. V indices: 2,3, 6,7.
    
    # This mapping is different from the 4x4x4 case.
    # Let's write a specific function for 4x4x2.
    
    def add_strassen_442(u_base, v_base, w_base):
        # u_base: start index for A in U (0)
        # v_base: start index for B in V (3 for cols 4-5, 6 for cols 7-8)
        # w_base: start index for C in W (0)
        
        # B1: rows 0-1, cols 0-1 of B.
        # In global V, these are v_base + 0, v_base + 1, v_base + 4, v_base + 5.
        # B2: rows 2-3, cols 0-1 of B.
        # In global V, these are v_base + 2, v_base + 3, v_base + 6, v_base + 7.
        
        # We can't use add_strassen_444 directly because the block structure is different.
        # We will manually construct the 4 terms.
        
        # Term 1: A11 * B1
        # A11 is block 0 of A. B1 is "block 0" of B (rows 0-1).
        # C1 is block 0 of C.
        
        # We need to map the 2x2x2 Strassen terms to these specific indices.
        
        # Let's define a helper to add a 2x2x2 Strassen product with custom index mapping.
        
        def add_222_custom(u_map, v_map, w_map):
            # u_map, v_map, w_map are dicts: local_idx -> global_idx
            for term in strassen_222:
                l_u_idx = term[0]
                l_u_val = term[1]
                l_v_idx = term[2]
                l_v_val = term[3]
                l_w_idx = term[4]
                l_w_val = term[5]
                
                g_u_idx = []
                g_u_val = []
                g_v_idx = []
                g_v_val = []
                g_w_idx = []
                g_w_val = []
                
                for li, val in zip(l_u_idx, l_u_val):
                    g_u_idx.append(u_map[li])
                    g_u_val.append(val)
                for li, val in zip(l_v_idx, l_v_val):
                    g_v_idx.append(v_map[li])
                    g_v_val.append(val)
                for li, val in zip(l_w_idx, l_w_val):
                    g_w_idx.append(w_map[li])
                    g_w_val.append(val)
                
                add_term(g_u_idx, g_u_val, g_v_idx, g_v_val, g_w_idx, g_w_val)
        
        # A11B1
        # A11: 0,1,4,5
        # B1: v_base+0, v_base+1, v_base+4, v_base+5
        # C1: 0,1,4,5
        u_map1 = {0: 0, 1: 1, 2: 4, 3: 5}
        v_map1 = {0: v_base, 1: v_base+1, 2: v_base+4, 3: v_base+5}
        w_map1 = {0: 0, 1: 1, 2: 4, 3: 5}
        add_222_custom(u_map1, v_map1, w_map1)
        
        # A12B2
        # A12: 2,3,6,7
        # B2: v_base+2, v_base+3, v_base+6, v_base+7
        # C1: 0,1,4,5
        u_map2 = {0: 2, 1: 3, 2: 6, 3: 7}
        v_map2 = {0: v_base+2, 1: v_base+3, 2: v_base+6, 3: v_base+7}
        w_map2 = {0: 0, 1: 1, 2: 4, 3: 5}
        add_222_custom(u_map2, v_map2, w_map2)
        
        # A21B1
        # A21: 8,9,12,13
        # B1: v_base+0, v_base+1, v_base+4, v_base+5
        # C2: 8,9,12,13
        u_map3 = {0: 8, 1: 9, 2: 12, 3: 13}
        v_map3 = {0: v_base, 1: v_base+1, 2: v_base+4, 3: v_base+5}
        w_map3 = {0: 8, 1: 9, 2: 12, 3: 13}
        add_222_custom(u_map3, v_map3, w_map3)
        
        # A22B2
        # A22: 10,11,14,15
        # B2: v_base+2, v_base+3, v_base+6, v_base+7
        # C2: 8,9,12,13
        u_map4 = {0: 10, 1: 11, 2: 14, 3: 15}
        v_map4 = {0: v_base+2, 1: v_base+3, 2: v_base+6, 3: v_base+7}
        w_map4 = {0: 8, 1: 9, 2: 12, 3: 13}
        add_222_custom(u_map4, v_map4, w_map4)
        
    # Add 4x4x2 for cols 7-8 (v_base = 6)
    add_strassen_442(0, 6, 0)
    
    # Add 4x4x2 for cols 5-6 (v_base = 4)
    add_strassen_442(0, 4, 0)
    
    # Add 4x4x1 for col 4 (v_base = 3)
    # Schoolbook 4x4x1
    # C_i = sum_j A_ij B_j
    # T_{i,j,k} = 1 if j=k? No.
    # C_i = sum_j A_ij B_{j,0}
    # So for each i, j: U[i*4+j, col] = 1, V[j*4+0, col] = 1, W[i, col] = 1?
    # No, W index is k*n+i. Here k=0. So W[i, col] = 1.
    
    for i in range(4):
        for j in range(4):
            add_term([i*4+j], [1], [j*4+3], [1], [i], [1])
            
    # Verify
    if time.time() < deadline - 5:
        def verify(U, V, W, n, m, p):
            nm = n * m
            mp = m * p
            pn = p * n
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1
                        actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(actual - expected) > 1e-9:
                            return False
            return True
        
        if not verify(U, V, W, n, m, p):
            raise ValueError("Decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
