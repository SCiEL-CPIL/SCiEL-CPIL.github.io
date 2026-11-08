import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    
    # Total rank
    rank_444 = 49
    rank_441 = 16
    rank = 2 * rank_444 + rank_441
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 tensors
    # u: A00, A01, A10, A11
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]), # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # P7
    ]
    
    # Macro terms for 4x4x4
    # Each term is defined by:
    # A_blocks: list of (block_index, sign)
    # B_blocks: list of (block_index, sign)
    # C_blocks: list of (block_index, sign)
    # The macro term is (sum A_blocks) * (sum B_blocks) -> sum C_blocks
    
    # M1: (A00+A11)(B00+B11) -> C00, C11?
    # Wait, the Strassen formulas for 4x4 are:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # Let's verify the C contributions.
    # M1 = (A00+A11)(B00+B11).
    # M1 is a 2x2 matrix.
    # In the 4x4 result, M1 appears in C00 and C11?
    # No, M1 is the product of two 2x2 matrices.
    # The result M1 is a 2x2 matrix.
    # How is M1 used?
    # C00 (top-left 2x2) = M1 - M4 + M6 - M7
    # C11 (bottom-right 2x2) = M1 - M3 + M2 - M5
    # So M1 contributes to C00 with +1 and to C11 with +1.
    
    macro_terms = [
        # M1
        {
            'A': [(0, 1), (3, 1)], # A00 + A11
            'B': [(0, 1), (3, 1)], # B00 + B11
            'C': [(0, 1), (3, 1)]  # C00 + C11
        },
        # M2
        {
            'A': [(2, 1), (3, 1)], # A10 + A11
            'B': [(0, 1)],         # B00
            'C': [(2, 1)]          # C10
        },
        # M3
        {
            'A': [(0, 1)],         # A00
            'B': [(1, 1), (3, -1)],# B01 - B11
            'C': [(1, 1)]          # C01
        },
        # M4
        {
            'A': [(3, 1)],         # A11
            'B': [(2, 1), (0, -1)],# B10 - B00
            'C': [(2, 1)]          # C10
        },
        # M5
        {
            'A': [(0, 1), (1, 1)], # A00 + A01
            'B': [(3, 1)],         # B11
            'C': [(3, 1)]          # C11
        },
        # M6
        {
            'A': [(2, 1), (0, -1)],# A10 - A00
            'B': [(0, 1), (1, 1)], # B00 + B01
            'C': [(0, 1)]          # C00
        },
        # M7
        {
            'A': [(1, 1), (3, -1)],# A01 - A11
            'B': [(2, 1), (3, 1)], # B10 + B11
            'C': [(0, 1)]          # C00
        }
    ]
    
    # Helper to add a 4x4x4 block decomposition
    # offset_b: offset for B columns (0 or 4)
    # offset_c: offset for C columns (0 or 4)
    # start_col: starting column index in U, V, W
    def add_444_block(offset_b, offset_c, start_col):
        col = start_col
        for term in macro_terms:
            # Expand A
            u_base = [0] * 16
            for block_idx, sign in term['A']:
                # Block idx 0: A[0:2, 0:2] -> indices 0,1,4,5
                # Block idx 1: A[0:2, 2:4] -> indices 2,3,6,7
                # Block idx 2: A[2:4, 0:2] -> indices 8,9,12,13
                # Block idx 3: A[2:4, 2:4] -> indices 10,11,14,15
                
                # Map block_idx to 4 scalar indices
                # Block 0: i=0,1; j=0,1. Indices: i*4+j. 0,1,4,5.
                # Block 1: i=0,1; j=2,3. Indices: 2,3,6,7.
                # Block 2: i=2,3; j=0,1. Indices: 8,9,12,13.
                # Block 3: i=2,3; j=2,3. Indices: 10,11,14,15.
                
                base_i = (block_idx // 2) * 2
                base_j = (block_idx % 2) * 2
                
                for di in range(2):
                    for dj in range(2):
                        idx = (base_i + di) * 4 + (base_j + dj)
                        u_base[idx] += sign
                        
            # Expand B
            v_base = [0] * 16
            for block_idx, sign in term['B']:
                base_i = (block_idx // 2) * 2
                base_j = (block_idx % 2) * 2
                for di in range(2):
                    for dj in range(2):
                        idx = (base_i + di) * 4 + (base_j + dj)
                        v_base[idx] += sign
                        
            # Expand C
            w_base = [0] * 16
            for block_idx, sign in term['C']:
                base_i = (block_idx // 2) * 2
                base_j = (block_idx % 2) * 2
                for di in range(2):
                    for dj in range(2):
                        idx = (base_i + di) * 4 + (base_j + dj)
                        w_base[idx] += sign
                        
            # Now apply the 2x2x2 Strassen tensors
            # The u_base, v_base, w_base are "block" vectors.
            # Each component of u_base corresponds to a 2x2 block.
            # Wait, u_base is length 16.
            # The Strassen tensor u' is length 4.
            # u'[k] corresponds to the k-th block.
            # So we need to combine u_base with u'.
            
            # Actually, the logic above is slightly off.
            # u_base is the sum of blocks.
            # The Strassen tensor u' is a linear combination of the 4 blocks.
            # So the final U vector is the sum of u'[k] * (block k vector).
            
            # Let's re-construct.
            # For a given macro term, we have linear forms for A', B', C'.
            # A' = sum sign_k * Block_k.
            # The Strassen tensor is u' (length 4) dot A' blocks.
            # So the final U vector (length 16) is:
            # U_final[i] = sum_{k=0..3} u'[k] * (Block_k_vector[i])
            
            for u_prime, v_prime, w_prime in strassen_222:
                u_final = [0] * 16
                v_final = [0] * 16
                w_final = [0] * 16
                
                for k in range(4):
                    if u_prime[k] != 0:
                        # Add u_prime[k] * Block_k
                        base_i = (k // 2) * 2
                        base_j = (k % 2) * 2
                        for di in range(2):
                            for dj in range(2):
                                idx = (base_i + di) * 4 + (base_j + dj)
                                u_final[idx] += u_prime[k]
                                
                for k in range(4):
                    if v_prime[k] != 0:
                        base_i = (k // 2) * 2
                        base_j = (k % 2) * 2
                        for di in range(2):
                            for dj in range(2):
                                idx = (base_i + di) * 4 + (base_j + dj)
                                v_final[idx] += v_prime[k]
                                
                for k in range(4):
                    if w_prime[k] != 0:
                        base_i = (k // 2) * 2
                        base_j = (k % 2) * 2
                        for di in range(2):
                            for dj in range(2):
                                idx = (base_i + di) * 4 + (base_j + dj)
                                w_final[idx] += w_prime[k]
                                
                # Now map to global U, V, W
                # U: same indices.
                for i in range(16):
                    if u_final[i] != 0:
                        U[i][col] = u_final[i]
                        
                # V: offset by offset_b
                # V index: j * 9 + k.
                # Local index in v_final: j * 4 + k.
                # Global k = k_local + offset_b.
                for j in range(4):
                    for k_local in range(4):
                        val = v_final[j * 4 + k_local]
                        if val != 0:
                            k_global = k_local + offset_b
                            v_idx = j * 9 + k_global
                            V[v_idx][col] = val
                            
                # W: offset by offset_c
                # W index: k * 4 + i.
                # Local index in w_final: k * 4 + i.
                # Global k = k_local + offset_c.
                for k_local in range(4):
                    for i in range(4):
                        val = w_final[k_local * 4 + i]
                        if val != 0:
                            k_global = k_local + offset_c
                            w_idx = k_global * 4 + i
                            W[w_idx][col] = val
                            
                col += 1
                
        return col
        
    # Add Block 1 (B cols 0-3, C cols 0-3)
    next_col = add_444_block(0, 0, 0)
    
    # Add Block 2 (B cols 4-7, C cols 4-7)
    next_col = add_444_block(4, 4, next_col)
    
    # Add Block 3 (B col 8, C col 8)
    # 16 terms
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][next_col] = 1
            V[j * 9 + 8][next_col] = 1
            W[8 * 4 + i][next_col] = 1
            next_col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
