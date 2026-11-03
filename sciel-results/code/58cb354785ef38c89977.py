import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns U (16 x rank), V (32 x rank), W (32 x rank) with integer/half-integer entries.
    Score is the rank.
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4
    strassen_2x2 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Construct 4x4x4 decomposition using Strassen
    # This will be a list of (u, v, w) where u, v, w are vectors of length 16
    # u corresponds to A (4x4), v to B (4x4), w to C (4x4)
    # Indices for 4x4: i, j in 0..3.
    # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # B blocks: B00 (0,1,4,5), B01 (2,3,6,7), B10 (8,9,12,13), B11 (10,11,14,15)
    # C blocks: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
    
    def expand_strassen_4x4(block_offset_b=0):
        """
        Returns list of (u, v, w) for 4x4x4 multiplication.
        block_offset_b is 0 for left half of B, 4 for right half (in terms of column block index for B)
        Actually, B is 4x4 here in the sub-problem.
        The offset applies to the global V and W indices.
        """
        decomps = []
        # Strassen terms for the 4 blocks
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Define block mappings
        # A blocks indices in u (length 16)
        # A00: 0,1,4,5; A01: 2,3,6,7; A10: 8,9,12,13; A11: 10,11,14,15
        # B blocks indices in v (length 16 relative to B start)
        # B00: 0,1,4,5; B01: 2,3,6,7; B10: 8,9,12,13; B11: 10,11,14,15
        # C blocks indices in w (length 16 relative to C start)
        # C00: 0,1,4,5; C01: 2,3,6,7; C10: 8,9,12,13; C11: 10,11,14,15
        
        # For each of the 7 Strassen multiplications, we have a 2x2x2 problem.
        # The "A" part of the 2x2 problem is a linear combination of A blocks.
        # The "B" part is a linear combination of B blocks.
        # The "C" part is a linear combination of C blocks.
        
        # We iterate over the 7 Strassen terms
        for s_idx in range(7):
            u_s, v_s, w_s = strassen_2x2[s_idx]
            
            # u_s is a vector of 4 coefficients for A00, A01, A10, A11
            # v_s is a vector of 4 coefficients for B00, B01, B10, B11
            # w_s is a vector of 4 coefficients for C00, C01, C10, C11
            
            # Now, each of these is a 2x2 matrix multiplication.
            # So we iterate over the 7 Strassen terms for the inner 2x2x2
            for inner_idx in range(7):
                u_i, v_i, w_i = strassen_2x2[inner_idx]
                
                # Construct full u, v, w vectors of length 16
                u_full = [0.0] * 16
                v_full = [0.0] * 16
                w_full = [0.0] * 16
                
                # Map A blocks
                # A00 at 0,1,4,5; A01 at 2,3,6,7; A10 at 8,9,12,13; A11 at 10,11,14,15
                # u_i is coefficients for the 2x2 elements of the "A" matrix in the inner Strassen step.
                # The "A" matrix in the outer Strassen step is formed by A blocks.
                # Wait, the outer Strassen step combines A blocks to form 2x2 matrices of blocks?
                # No, the outer Strassen step treats A00, A01, A10, A11 as scalars? No, they are matrices.
                # The multiplication M1 = (A00+A11)(B00+B11) is a multiplication of 2x2 matrices.
                # So u_s[0] is the coefficient for A00 in the first factor.
                # The first factor is a 2x2 matrix formed by A blocks.
                # Specifically, the entry (0,0) of the first factor is A00 * u_s[0]?
                # No.
                # Let's clarify the structure.
                # Outer Strassen:
                # M1 = (A00+A11)(B00+B11).
                # This is a matrix product of 2x2 matrices.
                # Let X = A00+A11 and Y = B00+B11.
                # X is 2x2, Y is 2x2.
                # The product Z = XY is 2x2.
                # Z contributes to C00 and C11?
                # In the 4x4 case:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # So M1 (which is a 2x2 matrix) contributes to C00 (with +1) and C11 (with +1).
                # So w_s[0]=1, w_s[3]=1.
                
                # Now, to decompose M1 = XY into rank-1 tensors:
                # We use the inner Strassen decomposition for 2x2x2.
                # The "A" input to the inner decomposition is X.
                # The "B" input is Y.
                # The "C" output is Z.
                
                # X is formed from A blocks:
                # X[0,0] = A00[0,0]*u_s[0] + A11[0,0]*u_s[3]?
                # No, u_s is [1, 0, 0, 1].
                # u_s[0] corresponds to A00. u_s[3] corresponds to A11.
                # So X = A00 * 1 + A11 * 1.
                # The elements of X are sums of elements of A00 and A11.
                
                # The inner Strassen u_i vector has 4 components corresponding to the 4 elements of X.
                # X[0,0] is at index 0, X[0,1] at 1, X[1,0] at 2, X[1,1] at 3.
                # So u_full will have non-zeros at the positions corresponding to A00 and A11 elements.
                
                # A00 elements are at global u indices: 0,1,4,5.
                # A11 elements are at global u indices: 10,11,14,15.
                
                # For each inner term k (0..6):
                # u_i[k] is the coefficient for the k-th element of X.
                # Element 0 of X is A00[0,0] + A11[0,0].
                # So u_full[0] += u_s[0] * u_i[0]
                # u_full[10] += u_s[3] * u_i[0]
                
                # General mapping:
                # Outer A block b (0..3) has offset in u:
                # b=0 (A00): 0,1,4,5
                # b=1 (A01): 2,3,6,7
                # b=2 (A10): 8,9,12,13
                # b=3 (A11): 10,11,14,15
                
                # Inner element e (0..3) corresponds to local index 0,1,2,3 in the block.
                # Local index 0 -> offset 0
                # Local index 1 -> offset 1
                # Local index 2 -> offset 4
                # Local index 3 -> offset 5
                
                # So for outer block b and inner element e:
                # global_u_idx = block_offset[b] + local_offset[e]
                
                block_offsets_u = [0, 2, 8, 10]
                local_offsets = [0, 1, 4, 5]
                
                # Similarly for V and W.
                # V blocks: B00, B01, B10, B11.
                # B is 4x4 in the sub-problem.
                # Global V indices for B_L: 0..15.
                # Global V indices for B_R: 16..31.
                # So we add block_offset_b * 16 to v_full indices.
                
                block_offsets_v = [0, 2, 8, 10]
                local_offsets_v = [0, 1, 4, 5]
                
                block_offsets_w = [0, 2, 8, 10]
                local_offsets_w = [0, 1, 4, 5]
                
                # Coefficients from outer Strassen
                # u_s[b] is coeff for A block b
                # v_s[b] is coeff for B block b
                # w_s[c] is coeff for C block c
                
                for b in range(4):
                    if u_s[b] == 0: continue
                    for e in range(4):
                        if u_i[e] == 0: continue
                        idx_u = block_offsets_u[b] + local_offsets[e]
                        u_full[idx_u] += u_s[b] * u_i[e]
                        
                for b in range(4):
                    if v_s[b] == 0: continue
                    for e in range(4):
                        if v_i[e] == 0: continue
                        idx_v = block_offsets_v[b] + local_offsets_v[e]
                        v_full[idx_v + block_offset_b * 16] += v_s[b] * v_i[e]
                        
                for c in range(4):
                    if w_s[c] == 0: continue
                    for e in range(4):
                        if w_i[e] == 0: continue
                        idx_w = block_offsets_w[c] + local_offsets_w[e]
                        w_full[idx_w + block_offset_b * 16] += w_s[c] * w_i[e]
                        
                decomps.append((u_full, v_full, w_full))
                
        return decomps

    # Get decompositions for left and right halves
    decomps_left = expand_strassen_4x4(0)
    decomps_right = expand_strassen_4x4(1)
    
    all_decomps = decomps_left + decomps_right
    rank = len(all_decomps)
    
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(32)]
    
    for l, (u, v, w) in enumerate(all_decomps):
        for i in range(16):
            U[i][l] = u[i]
        for j in range(32):
            V[j][l] = v[j]
        for k in range(32):
            W[k][l] = w[k]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
