"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition for 3x3x4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    # Strassen 2x2x2 decomposition
    # u, v, w are lists of (index, value) pairs
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(-1, 0), (2, 1)], [(0, 1), (2, 1)]), # v: -1 at 0, 1 at 2. w: 1 at 0, 1 at 2.
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # w: -1 at 0, 1 at 1.
        # M6
        ([(0, -1), (2, 1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in range(2):
        # Offset for B and C indices in this block
        # B is 3x4. Block 0 uses cols 0,1. Block 1 uses cols 2,3.
        # B index = j*p + k. p=4.
        # Block 0: k in 0,1. Block 1: k in 2,3.
        # C is 4x3. Block 0 uses rows 0,1. Block 1 uses rows 2,3.
        # C index = k*n + i. n=3.
        # Block 0: k in 0,1. Block 1: k in 2,3.
        
        b_k_offset = block * 2
        c_k_offset = block * 2
        
        # 1. Strassen block (7 terms)
        # A indices: 0,1,2,3 (top-left 2x2)
        # B indices: 0,1,2,3 (top 2x2 of this block) -> j in 0,1, k in 0,1
        # B global index: j*4 + (b_k_offset + k_local)
        # C indices: 0,1,2,3 (top 2x2 of this block) -> i in 0,1, k in 0,1
        # C global index: (c_k_offset + k_local)*3 + i
        
        for u_list, v_list, w_list in strassen_terms:
            for idx, val in u_list:
                U[idx][col] += val
            for idx, val in v_list:
                # idx is local B index (0..3)
                # global B index = idx // 2 * 4 + (b_k_offset + idx % 2)
                j_loc = idx // 2
                k_loc = idx % 2
                g_idx = j_loc * 4 + (b_k_offset + k_loc)
                V[g_idx][col] += val
            for idx, val in w_list:
                # idx is local C index (0..3)
                # global C index = (c_k_offset + k_loc) * 3 + i_loc
                i_loc = idx % 2
                k_loc = idx // 2
                g_idx = (c_k_offset + k_loc) * 3 + i_loc
                W[g_idx][col] += val
            col += 1
            
        # 2. A12 * B21 block (4 terms)
        # A12: A[0,2], A[1,2] -> indices 2, 5
        # B21: B[2, b_k_offset], B[2, b_k_offset+1] -> indices 2*4 + b_k_offset, 2*4 + b_k_offset + 1
        # C12: C[c_k_offset, 1], C[c_k_offset+1, 1] -> indices (c_k_offset)*3 + 1, (c_k_offset+1)*3 + 1
        
        a_02 = 2
        a_12 = 5
        b_20 = 8 + b_k_offset
        b_21 = 8 + b_k_offset + 1
        c_01 = c_k_offset * 3 + 1
        c_11 = (c_k_offset + 1) * 3 + 1
        
        add_term(a_02, 1, b_20, 1, c_01, 1, col); col += 1
        add_term(a_02, 1, b_21, 1, c_01, 1, col); col += 1 # Wait, C01 gets A02*B21?
        # C_top = A11 B_top + A12 B_bottom
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
        # So A[0,2]B[2,1] contributes to C[0,1].
        # My previous analysis:
        # A[0,2] B[2,0] -> C[0,0]
        # A[0,2] B[2,1] -> C[0,1]
        # A[1,2] B[2,0] -> C[1,0]
        # A[1,2] B[2,1] -> C[1,1]
        
        # Let's re-verify indices for C12 block
        # C[0,1] is index 1 in local C (row 0, col 1). Global index c_k_offset*3 + 1.
        # C[1,1] is index 3 in local C (row 1, col 1). Global index (c_k_offset+1)*3 + 1.
        
        # Correction:
        # Term 1: A[0,2] B[2,0] -> C[0,0] (local idx 0, global c_k_offset*3)
        # Term 2: A[0,2] B[2,1] -> C[0,1] (local idx 1, global c_k_offset*3 + 1)
        # Term 3: A[1,2] B[2,0] -> C[1,0] (local idx 2, global (c_k_offset+1)*3)
        # Term 4: A[1,2] B[2,1] -> C[1,1] (local idx 3, global (c_k_offset+1)*3 + 1)
        
        # I made a mistake in the add_term calls above. Let's rewrite this section properly.
        col -= 4 # rollback
        c_00 = c_k_offset * 3
        c_10 = (c_k_offset + 1) * 3
        
        add_term(a_02, 1, b_20, 1, c_00, 1, col); col += 1
        add_term(a_02, 1, b_21, 1, c_01, 1, col); col += 1
        add_term(a_12, 1, b_20, 1, c_10, 1, col); col += 1
        add_term(a_12, 1, b_21, 1, c_11, 1, col); col += 1
        
        # 3. A21 * B_top block (4 terms)
        # A21: A[2,0], A[2,1] -> indices 6, 7
        # B_top: B[0, b_k_offset], B[0, b_k_offset+1], B[1, b_k_offset], B[1, b_k_offset+1]
        # B indices: 0*4 + b_k_offset, 0*4 + b_k_offset + 1, 1*4 + b_k_offset, 1*4 + b_k_offset + 1
        # C_bottom: C[c_k_offset, 0], C[c_k_offset, 1], C[c_k_offset+1, 0], C[c_k_offset+1, 1]
        # C indices: c_k_offset*3, c_k_offset*3+1, (c_k_offset+1)*3, (c_k_offset+1)*3+1
        
        a_20 = 6
        a_21 = 7
        b_00 = b_k_offset
        b_01 = b_k_offset + 1
        b_10 = 4 + b_k_offset
        b_11 = 4 + b_k_offset + 1
        
        c_20 = c_k_offset * 3
        c_21 = c_k_offset * 3 + 1
        c_30 = (c_k_offset + 1) * 3
        c_31 = (c_k_offset + 1) * 3 + 1
        
        add_term(a_20, 1, b_00, 1, c_20, 1, col); col += 1
        add_term(a_20, 1, b_01, 1, c_21, 1, col); col += 1
        add_term(a_21, 1, b_10, 1, c_20, 1, col); col += 1
        add_term(a_21, 1, b_11, 1, c_21, 1, col); col += 1
        
        # 4. A22 * B_bottom block (2 terms)
        # A22: A[2,2] -> index 8
        # B_bottom: B[2, b_k_offset], B[2, b_k_offset+1] -> indices 8 + b_k_offset, 8 + b_k_offset + 1
        # C_bottom: C[c_k_offset, 0], C[c_k_offset, 1] -> indices c_k_offset*3, c_k_offset*3+1
        # Note: C_bottom in this block is row 2 of C? No, C is 4x3.
        # Block 0: C rows 0,1. Block 1: C rows 2,3.
        # So C_bottom for block 0 is rows 0,1? No.
        # My block decomposition of C:
        # C_top (2x2) corresponds to A11, A12.
        # C_bottom (1x2) corresponds to A21, A22.
        # Wait, C is 3x2 in the 3x3x2 case.
        # C_top is 2x2 (rows 0,1). C_bottom is 1x2 (row 2).
        # In the 3x3x4 case, we split C into two 3x2 blocks.
        # Block 0: C cols 0,1. Block 1: C cols 2,3.
        # So for each block, we have a 3x2 C matrix.
        # The "C_bottom" of that 3x2 matrix is row 2.
        # So for Block 0, C_bottom is C[2,0], C[2,1]. Indices 6,7.
        # For Block 1, C_bottom is C[2,2], C[2,3]. Indices 8,9.
        
        # My previous c_k_offset logic:
        # c_k_offset = block * 2.
        # Block 0: c_k_offset = 0. C rows 0,1,2? No.
        # In 3x3x2, C is 3x2. Indices 0..5.
        # Row 0: 0,1. Row 1: 2,3. Row 2: 4,5.
        # My Strassen block used C indices 0..3 (Rows 0,1).
        # My A12 block used C indices 1,3 (Col 1 of Rows 0,1).
        # My A21 block used C indices 4,5 (Row 2).
        # My A22 block used C indices 4,5 (Row 2).
        
        # So for Block 0 (k=0,1), C indices are 0..5.
        # For Block 1 (k=2,3), C indices are 6..11.
        # C index = k*n + i.
        # Block 0: k in 0,1. Indices 0..5.
        # Block 1: k in 2,3. Indices 6..11.
        
        # So c_k_offset should be block * 2 * n? No.
        # c_k_offset is the offset in k.
        # Global C index = (k_local + block*2)*3 + i_local.
        
        # Let's re-verify A21 and A22 C indices.
        # A21 block: C row 2.
        # Block 0: C[2,0], C[2,1]. Indices 2*3+0=6, 2*3+1=7.
        # Block 1: C[2,2], C[2,3]. Indices 2*3+2=8, 2*3+3=9.
        
        # My previous c_20, c_21 etc were wrong for Block 1.
        # c_k_offset = block * 2.
        # c_20 = (c_k_offset + 2) * 3 + 0? No.
        # Row 2 of C. k = 2 + block*2? No.
        # In 3x3x2, C is 3x2. Rows 0,1,2.
        # Block 0 uses C rows 0,1,2? No, C is 3x2.
        # The 3x3x2 tensor has C dimension 3x2.
        # So for each block of B (2 cols), we have a full 3x2 C matrix.
        # So Block 0 C indices: 0..5. Block 1 C indices: 6..11.
        
        # Strassen block: C rows 0,1.
        # Block 0: 0..3. Block 1: 6..9.
        # A12 block: C col 1, rows 0,1.
        # Block 0: 1, 3. Block 1: 7, 9.
        # A21 block: C row 2, cols 0,1.
        # Block 0: 4, 5. Block 1: 10, 11.
        # A22 block: C row 2, cols 0,1.
        # Block 0: 4, 5. Block 1: 10, 11.
        
        # Let's fix the indices for A21 and A22.
        col -= 6 # rollback A21 and A22 (6 terms)
        
        c_20 = (block * 2 + 2) * 3 + 0
        c_21 = (block * 2 + 2) * 3 + 1
        
        add_term(a_20, 1, b_00, 1, c_20, 1, col); col += 1
        add_term(a_20, 1, b_01, 1, c_21, 1, col); col += 1
        add_term(a_21, 1, b_10, 1, c_20, 1, col); col += 1
        add_term(a_21, 1, b_11, 1, c_21, 1, col); col += 1
        
        a_22 = 8
        add_term(a_22, 1, b_20, 1, c_20, 1, col); col += 1
        add_term(a_22, 1, b_21, 1, c_21, 1, col); col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
