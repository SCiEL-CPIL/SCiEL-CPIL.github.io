import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # U: (16, rank), V: (32, rank), W: (32, rank)
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M0 = (q00 + q11)(r00 + r11)
    # M1 = (q10 + q11)r00
    # M2 = q00(r01 - r11)
    # M3 = q11(r10 - r00)
    # M4 = (q00 + q01)r11
    # M5 = (q10 - q00)(r00 + r01)
    # M6 = (q01 - q11)(r10 + r11)
    
    # Inner Strassen: computes P = Q R
    # Returns 7 pairs of (u_vec, v_vec) where u is from Q, v is from R
    # Q indices: 0:00, 1:01, 2:10, 3:11
    # R indices: 0:00, 1:01, 2:10, 3:11
    
    def get_inner_strassen_terms(q_indices, r_indices):
        # q_indices: dict mapping local 2x2 index to global A index
        # r_indices: dict mapping local 2x2 index to global B index
        # Returns list of 7 tuples (u, v)
        # u is a dict of {global_A_index: coeff}
        # v is a dict of {global_B_index: coeff}
        
        terms = []
        
        # M0
        u0 = {q_indices[0]: 1, q_indices[3]: 1}
        v0 = {r_indices[0]: 1, r_indices[3]: 1}
        terms.append((u0, v0))
        
        # M1
        u1 = {q_indices[2]: 1, q_indices[3]: 1}
        v1 = {r_indices[0]: 1}
        terms.append((u1, v1))
        
        # M2
        u2 = {q_indices[0]: 1}
        v2 = {r_indices[1]: 1, r_indices[3]: -1}
        terms.append((u2, v2))
        
        # M3
        u3 = {q_indices[3]: 1}
        v3 = {r_indices[2]: 1, r_indices[0]: -1}
        terms.append((u3, v3))
        
        # M4
        u4 = {q_indices[0]: 1, q_indices[1]: 1}
        v4 = {r_indices[3]: 1}
        terms.append((u4, v4))
        
        # M5
        u5 = {q_indices[2]: 1, q_indices[0]: -1}
        v5 = {r_indices[0]: 1, r_indices[1]: 1}
        terms.append((u5, v5))
        
        # M6
        u6 = {q_indices[1]: 1, q_indices[3]: -1}
        v6 = {r_indices[2]: 1, r_indices[3]: 1}
        terms.append((u6, v6))
        
        return terms

    # Outer Strassen: computes C = A B
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # Returns 7 pairs of (X_dict, Y_dict)
    # X_dict maps local 2x2 index of X to global A index
    # Y_dict maps local 2x2 index of Y to global B index
    
    def get_outer_strassen_terms(a_block_indices, b_block_indices):
        # a_block_indices: dict mapping block (00,01,10,11) to list of 4 global A indices
        # b_block_indices: dict mapping block (00,01,10,11) to list of 4 global B indices
        
        terms = []
        
        # M0 = (A00 + A11)(B00 + B11)
        x0 = {}
        for i in range(4):
            x0[a_block_indices['00'][i]] = 1
            x0[a_block_indices['11'][i]] = 1
        y0 = {}
        for i in range(4):
            y0[b_block_indices['00'][i]] = 1
            y0[b_block_indices['11'][i]] = 1
        terms.append((x0, y0))
        
        # M1 = (A10 + A11)B00
        x1 = {}
        for i in range(4):
            x1[a_block_indices['10'][i]] = 1
            x1[a_block_indices['11'][i]] = 1
        y1 = {}
        for i in range(4):
            y1[b_block_indices['00'][i]] = 1
        terms.append((x1, y1))
        
        # M2 = A00(B01 - B11)
        x2 = {}
        for i in range(4):
            x2[a_block_indices['00'][i]] = 1
        y2 = {}
        for i in range(4):
            y2[b_block_indices['01'][i]] = 1
            y2[b_block_indices['11'][i]] = -1
        terms.append((x2, y2))
        
        # M3 = A11(B10 - B00)
        x3 = {}
        for i in range(4):
            x3[a_block_indices['11'][i]] = 1
        y3 = {}
        for i in range(4):
            y3[b_block_indices['10'][i]] = 1
            y3[b_block_indices['00'][i]] = -1
        terms.append((x3, y3))
        
        # M4 = (A00 + A01)B11
        x4 = {}
        for i in range(4):
            x4[a_block_indices['00'][i]] = 1
            x4[a_block_indices['01'][i]] = 1
        y4 = {}
        for i in range(4):
            y4[b_block_indices['11'][i]] = 1
        terms.append((x4, y4))
        
        # M5 = (A10 - A00)(B00 + B01)
        x5 = {}
        for i in range(4):
            x5[a_block_indices['10'][i]] = 1
            x5[a_block_indices['00'][i]] = -1
        y5 = {}
        for i in range(4):
            y5[b_block_indices['00'][i]] = 1
            y5[b_block_indices['01'][i]] = 1
        terms.append((x5, y5))
        
        # M6 = (A01 - A11)(B10 + B11)
        x6 = {}
        for i in range(4):
            x6[a_block_indices['01'][i]] = 1
            x6[a_block_indices['11'][i]] = -1
        y6 = {}
        for i in range(4):
            y6[b_block_indices['10'][i]] = 1
            y6[b_block_indices['11'][i]] = 1
        terms.append((x6, y6))
        
        return terms

    # Helper to get global indices for a block
    # Block (br, bc) of A: i in [2*br, 2*br+1], j in [2*bc, 2*bc+1]
    # Global index i*m + j
    
    def get_a_block_indices(br, bc):
        indices = []
        for i in range(2*br, 2*br+2):
            for j in range(2*bc, 2*bc+2):
                indices.append(i * m + j)
        return indices

    # Block (br, bc) of B: j in [2*br, 2*br+1], k in [k_offset + 2*bc, k_offset + 2*bc+1]
    # Global index j*p + k
    
    def get_b_block_indices(br, bc, k_offset):
        indices = []
        for j in range(2*br, 2*br+2):
            for k in range(k_offset + 2*bc, k_offset + 2*bc+2):
                indices.append(j * p + k)
        return indices

    # Inner Strassen contribution to P entries
    # P00 = M0 + M3 - M4 + M6
    # P01 = M2 + M4
    # P10 = M1 + M3
    # P11 = M0 - M1 + M2 + M5
    
    inner_contrib = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # P00
        {2: 1, 4: 1},               # P01
        {1: 1, 3: 1},               # P10
        {0: 1, 1: -1, 2: 1, 5: 1}  # P11
    ]
    
    # Outer Strassen contribution to C blocks
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    outer_contrib = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]

    col = 0
    for half in range(2):
        k_offset = half * 4
        
        # Define block indices for A
        a_blocks = {
            '00': get_a_block_indices(0, 0),
            '01': get_a_block_indices(0, 1),
            '10': get_a_block_indices(1, 0),
            '11': get_a_block_indices(1, 1)
        }
        
        # Define block indices for B (for this half)
        b_blocks = {
            '00': get_b_block_indices(0, 0, k_offset),
            '01': get_b_block_indices(0, 1, k_offset),
            '10': get_b_block_indices(1, 0, k_offset),
            '11': get_b_block_indices(1, 1, k_offset)
        }
        
        outer_terms = get_outer_strassen_terms(a_blocks, b_blocks)
        
        for s in range(7):
            x_dict, y_dict = outer_terms[s]
            
            # x_dict maps global A index to coeff (for the 2x2 matrix X_s)
            # But we need to map to local 2x2 indices for inner Strassen
            # X_s is a 2x2 matrix.
            # x_dict contains 8 entries (4 from one block, 4 from another).
            # We need to group them into local 2x2 indices.
            
            # Local indices for X_s:
            # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
            # The global indices in x_dict are from two A blocks.
            # We need to know which local index each global index corresponds to.
            
            # Let's reconstruct the local index mapping for X_s
            # X_s is sum of two A blocks.
            # For each global index in x_dict, we can determine its local position.
            
            q_indices = {} # local_idx -> global_idx
            # We need to iterate through the blocks used in x_dict
            
            # This is tricky because x_dict is a flat dict.
            # Let's just iterate through all possible global A indices and check if they are in x_dict.
            # There are 16 global A indices.
            
            for local_i in range(2):
                for local_j in range(2):
                    # Find the global index corresponding to this local position in X_s
                    # X_s is formed by adding two A blocks.
                    # The local position (local_i, local_j) in X_s corresponds to
                    # the same position in the source A blocks.
                    
                    # We need to identify which A blocks were added.
                    # Let's re-derive the structure.
                    
                    # Instead of generic dicts, let's explicitly define the 7 outer terms
                    # with their block compositions.
                    
                    pass

    # The generic approach above is getting messy with index mapping.
    # Let's hardcode the 7 outer terms and their block compositions.
    
    # Term 0: (A00 + A11)(B00 + B11)
    # Term 1: (A10 + A11)B00
    # Term 2: A00(B01 - B11)
    # Term 3: A11(B10 - B00)
    # Term 4: (A00 + A01)B11
    # Term 5: (A10 - A00)(B00 + B01)
    # Term 6: (A01 - A11)(B10 + B11)
    
    # For each term, we define:
    # x_blocks: list of (block_name, sign)
    # y_blocks: list of (block_name, sign)
    
    outer_struct = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ([('10', 1), ('11', 1)], [('00', 1)]),
        ([('00', 1)], [('01', 1), ('11', -1)]),
        ([('11', 1)], [('10', 1), ('00', -1)]),
        ([('00', 1), ('01', 1)], [('11', 1)]),
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)])
    ]
    
    col = 0
    for half in range(2):
        k_offset = half * 4
        
        a_blocks = {
            '00': get_a_block_indices(0, 0),
            '01': get_a_block_indices(0, 1),
            '10': get_a_block_indices(1, 0),
            '11': get_a_block_indices(1, 1)
        }
        
        b_blocks = {
            '00': get_b_block_indices(0, 0, k_offset),
            '01': get_b_block_indices(0, 1, k_offset),
            '10': get_b_block_indices(1, 0, k_offset),
            '11': get_b_block_indices(1, 1, k_offset)
        }
        
        for s in range(7):
            x_struct, y_struct = outer_struct[s]
            
            # Construct q_indices and r_indices for inner Strassen
            # q_indices[local_idx] = global_A_index
            # We need to map local 2x2 positions to global indices.
            
            # For X_s, it is a sum of A blocks.
            # Local position (li, lj) in X_s corresponds to position (li, lj) in each contributing A block.
            # So we can just pick one of the contributing blocks to get the global index,
            # because the local structure is the same.
            
            # Wait, if X_s = A00 + A11, then X_s[0,0] = A00[0,0] + A11[0,0].
            # The global index of A00[0,0] is 0.
            # The global index of A11[0,0] is 8.
            # So X_s[0,0] involves global indices 0 and 8.
            
            # The inner Strassen algorithm takes q_indices[0..3] as the 4 entries of the 2x2 matrix Q.
            # But here, each entry of Q is a SUM of global A entries.
            # So we can't just pass a single global index.
            # We need to pass a dict of {global_idx: coeff} for each local position.
            
            # Let's modify get_inner_strassen_terms to accept dicts of dicts.
            
            q_entries = [{} for _ in range(4)]
            r_entries = [{} for _ in range(4)]
            
            for block_name, sign in x_struct:
                indices = a_blocks[block_name]
                for local_pos in range(4):
                    q_entries[local_pos][indices[local_pos]] += sign
            
            for block_name, sign in y_struct:
                indices = b_blocks[block_name]
                for local_pos in range(4):
                    r_entries[local_pos][indices[local_pos]] += sign
            
            # Now call inner Strassen with these entries
            inner_terms = get_inner_strassen_terms(q_entries, r_entries)
            
            for t in range(7):
                u_dict, v_dict = inner_terms[t]
                
                # u_dict is {global_A_idx: coeff}
                # v_dict is {global_B_idx: coeff}
                
                for a_idx, coeff in u_dict.items():
                    U[a_idx][col] += coeff
                
                for b_idx, coeff in v_dict.items():
                    V[b_idx][col] += coeff
                
                # Compute W contributions
                # For each output index r (0..31)
                for r in range(p * n):
                    k = r // n
                    i = r % n
                    
                    # Determine block indices
                    # k is 0..7
                    # i is 0..3
                    
                    # Block of k:
                    # If half=0, k is 0..3. Block B = k // 2.
                    # If half=1, k is 4..7. Block B = (k-4) // 2.
                    # But wait, the outer Strassen was defined for 4x4x4.
                    # So for half=0, k_local = k.
                    # For half=1, k_local = k - 4.
                    
                    k_local = k - k_offset
                    B = k_local // 2
                    A = i // 2
                    
                    # Local indices within the block
                    u_local = i % 2
                    v_local = k_local % 2
                    
                    # Get outer sign
                    if s not in outer_contrib[A * 2 + B]:
                        w_coeff = 0
                    else:
                        w_coeff = outer_contrib[A * 2 + B][s]
                    
                    if w_coeff != 0:
                        # Get inner coefficient
                        # inner_contrib[v_local * 2 + u_local]
                        # Wait, C is (k, i). So row k, col i.
                        # In our matrix multiplication C = A B, C[k, i] = sum A[i, j] B[j, k].
                        # The task says W[k*n+i, r].
                        # So W index is k*n + i.
                        # My r loop is over p*n indices.
                        # k = r // n, i = r % n. Correct.
                        
                        # The inner Strassen computes P = Q R.
                        # P is 2x2.
                        # P[u_local, v_local] corresponds to C block entry.
                        # Wait, in Strassen, P is the result of Q R.
                        # Q is from A, R is from B.
                        # So P[u, v] is the (u, v) entry of the product.
                        # This corresponds to C[u, v] in the local block.
                        # So we need inner_contrib for (u_local, v_local).
                        
                        # My inner_contrib list is indexed by [row][col]?
                        # I defined:
                        # inner_contrib[0] -> P00
                        # inner_contrib[1] -> P01
                        # inner_contrib[2] -> P10
                        # inner_contrib[3] -> P11
                        
                        # So index = u_local * 2 + v_local.
                        
                        inner_idx = u_local * 2 + v_local
                        if t in inner_contrib[inner_idx]:
                            w_coeff *= inner_contrib[inner_idx][t]
                    
                    W[r][col] = w_coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
