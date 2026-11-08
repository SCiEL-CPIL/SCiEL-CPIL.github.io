import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank for 4x4x4 is 49 (Strassen squared)
    # Rank for 4x4x1 is 4
    rank_block_444 = 49
    rank_block_441 = 4
    
    total_rank = 2 * rank_block_444 + rank_block_441
    R = total_rank
    
    # Initialize U, V, W
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (36, R)
    # W: (p*n, R) = (36, R)
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to construct 2x2x2 Strassen factors
    # Returns U2 (4x7), V2 (4x7), W2 (4x7)
    # Indices:
    # U2[i*2+j, r]
    # V2[j*2+k, r]
    # W2[k*2+i, r]
    def strassen_2x2x2():
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        # Standard Strassen for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U2 factors (A side)
        # Col 0 (M1): A11+A22 -> U2[0,0]=1, U2[3,0]=1
        # Col 1 (M2): A21+A22 -> U2[2,1]=1, U2[3,1]=1
        # Col 2 (M3): A11 -> U2[0,2]=1
        # Col 3 (M4): A22 -> U2[3,3]=1
        # Col 4 (M5): A11+A12 -> U2[0,4]=1, U2[1,4]=1
        # Col 5 (M6): A21-A11 -> U2[2,5]=1, U2[0,5]=-1
        # Col 6 (M7): A12-A22 -> U2[1,6]=1, U2[3,6]=-1
        
        U2[0, 0] = 1; U2[3, 0] = 1
        U2[2, 1] = 1; U2[3, 1] = 1
        U2[0, 2] = 1
        U2[3, 3] = 1
        U2[0, 4] = 1; U2[1, 4] = 1
        U2[2, 5] = 1; U2[0, 5] = -1
        U2[1, 6] = 1; U2[3, 6] = -1
        
        # V2 factors (B side)
        # Col 0 (M1): B11+B22 -> V2[0,0]=1, V2[3,0]=1
        # Col 1 (M2): B11 -> V2[0,1]=1
        # Col 2 (M3): B12-B22 -> V2[1,2]=1, V2[3,2]=-1
        # Col 3 (M4): B21-B11 -> V2[2,3]=1, V2[0,3]=-1
        # Col 4 (M5): B22 -> V2[3,4]=1
        # Col 5 (M6): B11+B12 -> V2[0,5]=1, V2[1,5]=1
        # Col 6 (M7): B21+B22 -> V2[2,6]=1, V2[3,6]=1
        
        V2[0, 0] = 1; V2[3, 0] = 1
        V2[0, 1] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        V2[2, 3] = 1; V2[0, 3] = -1
        V2[3, 4] = 1
        V2[0, 5] = 1; V2[1, 5] = 1
        V2[2, 6] = 1; V2[3, 6] = 1
        
        # W2 factors (C side)
        # Col 0 (M1): C11 + C22 -> W2[0,0]=1, W2[3,0]=1
        # Col 1 (M2): C21 -> W2[2,1]=1
        # Col 2 (M3): C12 -> W2[1,2]=1
        # Col 3 (M4): C11 + C21 -> W2[0,3]=1, W2[2,3]=1
        # Col 4 (M5): C11 - C22 -> W2[0,4]=1, W2[3,4]=-1
        # Col 5 (M6): C22 -> W2[3,5]=1
        # Col 6 (M7): C11 + C12 -> W2[0,6]=1, W2[1,6]=1
        
        W2[0, 0] = 1; W2[3, 0] = 1
        W2[2, 1] = 1
        W2[1, 2] = 1
        W2[0, 3] = 1; W2[2, 3] = 1
        W2[0, 4] = 1; W2[3, 4] = -1
        W2[3, 5] = 1
        W2[0, 6] = 1; W2[1, 6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2x2()
    
    # Construct 4x4x4 Strassen factors using Kronecker logic
    # We treat 4x4 as 2x2 blocks of 2x2.
    # Block indices (bi, bj) for A, (bj, bk) for B, (bk, bi) for C.
    # Element indices (ei, ej) for A, (ej, ek) for B, (ek, ei) for C.
    
    U444 = np.zeros((16, 49), dtype=int)
    V444 = np.zeros((36, 49), dtype=int)
    W444 = np.zeros((36, 49), dtype=int)
    
    # U2 indices: i_blk * 2 + i_el, j_blk * 2 + j_el ? No.
    # U2 is for 2x2x2.
    # In U444, row index is i * 4 + j.
    # i = i_blk * 2 + i_el. j = j_blk * 2 + j_el.
    # So row = (i_blk * 2 + i_el) * 4 + (j_blk * 2 + j_el)
    #        = i_blk * 8 + i_el * 4 + j_blk * 2 + j_el.
    
    # V2 indices: j * 2 + k.
    # In V444, row index is j * 9 + k? No, for the block, it's 4x4.
    # But we are building the factor for the 4x4x4 block itself.
    # The global V will map these to the full 4x9.
    # For the 4x4x4 block, V rows are j * 4 + k.
    # j = j_blk * 2 + j_el. k = k_blk * 2 + k_el.
    # Row = (j_blk * 2 + j_el) * 4 + (k_blk * 2 + k_el)
    #      = j_blk * 8 + j_el * 4 + k_blk * 2 + k_el.
    
    # W2 indices: k * 2 + i.
    # In W444, row index is k * 4 + i.
    # k = k_blk * 2 + k_el. i = i_blk * 2 + i_el.
    # Row = (k_blk * 2 + k_el) * 4 + (i_blk * 2 + i_el)
    #      = k_blk * 8 + k_el * 4 + i_blk * 2 + i_el.
    
    # The rank 49 comes from 7 block ranks and 7 element ranks.
    # Col index r = r_blk * 7 + r_el.
    
    for r_blk in range(7):
        for r_el in range(7):
            r = r_blk * 7 + r_el
            # Block level factors
            # U_blk has indices (i_blk, j_blk)
            # V_blk has indices (j_blk, k_blk)
            # W_blk has indices (k_blk, i_blk)
            # We need to extract the specific block-level entries from U2, V2, W2?
            # No, U2 is the factor for the 2x2 matrix itself.
            # The "block level" Strassen is applied to the 2x2 grid of blocks.
            # So the "U2" used for block level is the same structure, but its
            # "elements" are 2x2 matrices.
            # The factor value for a specific block configuration is the same as U2.
            
            # For a given (i_blk, j_blk, r_blk), the contribution to U444 is
            # U2[i_blk * 2 + j_blk, r_blk] (Wait, U2 indices for 2x2 are i*2+j).
            # Yes.
            
            u_blk_val = U2[i_blk * 2 + j_blk, r_blk] if False else 0 # Placeholder logic
            
            # Iterate over all block and element indices
            for i_blk in range(2):
                for j_blk in range(2):
                    u_blk = U2[i_blk * 2 + j_blk, r_blk]
                    if u_blk == 0: continue
                    for k_blk in range(2):
                        v_blk = V2[j_blk * 2 + k_blk, r_blk]
                        if v_blk == 0: continue
                        w_blk = W2[k_blk * 2 + i_blk, r_blk]
                        if w_blk == 0: continue
                        
                        # Now iterate over element indices
                        for i_el in range(2):
                            for j_el in range(2):
                                u_el = U2[i_el * 2 + j_el, r_el]
                                if u_el == 0: continue
                                # Full U index
                                u_row = (i_blk * 2 + i_el) * 4 + (j_blk * 2 + j_el)
                                U444[u_row, r] += u_blk * u_el
                                
                                for k_el in range(2):
                                    v_el = V2[j_el * 2 + k_el, r_el]
                                    if v_el == 0: continue
                                    w_el = W2[k_el * 2 + i_el, r_el]
                                    if w_el == 0: continue
                                    
                                    v_row = (j_blk * 2 + j_el) * 4 + (k_blk * 2 + k_el)
                                    V444[v_row, r] += v_blk * v_el
                                    
                                    w_row = (k_blk * 2 + k_el) * 4 + (i_blk * 2 + i_el)
                                    W444[w_row, r] += w_blk * w_el
                                    
    # Now assemble global U, V, W
    
    # Block 1: Cols 0-3 of B.
    # U is same for both 4x4 blocks.
    # V rows for B1: j * 9 + k, where j in 0..3, k in 0..3.
    # V444 rows are j * 4 + k.
    # We need to map V444[row, col] to V[row_mapped, col_global].
    
    # Map V444 (16 rows) to V (36 rows) for B1
    for j in range(4):
        for k in range(4):
            v444_row = j * 4 + k
            v_global_row = j * 9 + k
            for r in range(49):
                V[v_global_row, r] = V444[v444_row, r]
                
    # Map U444 (16 rows) to U (16 rows)
    # U is the same for both blocks, so we just copy.
    for i in range(4):
        for j in range(4):
            u444_row = i * 4 + j
            u_global_row = i * 4 + j
            for r in range(49):
                U[u_global_row, r] = U444[u444_row, r]
                
    # Map W444 (16 rows) to W (36 rows) for C1 (cols 0-3)
    # W444 rows are k * 4 + i.
    # W global rows for C1 are k * 4 + i (since k is 0..3, n=4).
    # Wait, W index is k * n + i.
    # For C1, k is 0..3.
    for k in range(4):
        for i in range(4):
            w444_row = k * 4 + i
            w_global_row = k * 4 + i
            for r in range(49):
                W[w_global_row, r] = W444[w444_row, r]
                
    # Block 2: Cols 4-7 of B.
    # U is same.
    # V rows for B2: j * 9 + (4 + k).
    for j in range(4):
        for k in range(4):
            v444_row = j * 4 + k
            v_global_row = j * 9 + (4 + k)
            for r in range(49):
                # Note: U is already filled. We don't add to U again?
                # The rank is additive.
                # So for the second block, we use columns 49..97.
                pass
                
    # Correct Assembly:
    # Columns 0-48: Block 1
    # Columns 49-97: Block 2
    # Columns 98-101: Block 3 (4x4x1)
    
    # Reset and do properly
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Block 1
    base_r = 0
    for i in range(4):
        for j in range(4):
            u444_row = i * 4 + j
            u_global_row = i * 4 + j
            for r in range(49):
                U[u_global_row][base_r + r] = U444[u444_row][r]
                
    for j in range(4):
        for k in range(4):
            v444_row = j * 4 + k
            v_global_row = j * 9 + k
            for r in range(49):
                V[v_global_row][base_r + r] = V444[v444_row][r]
                
    for k in range(4):
        for i in range(4):
            w444_row = k * 4 + i
            w_global_row = k * 4 + i
            for r in range(49):
                W[w_global_row][base_r + r] = W444[w444_row][r]
                
    # Block 2
    base_r = 49
    for i in range(4):
        for j in range(4):
            u444_row = i * 4 + j
            u_global_row = i * 4 + j
            for r in range(49):
                U[u_global_row][base_r + r] = U444[u444_row][r]
                
    for j in range(4):
        for k in range(4):
            v444_row = j * 4 + k
            v_global_row = j * 9 + (4 + k)
            for r in range(49):
                V[v_global_row][base_r + r] = V444[v444_row][r]
                
    for k in range(4):
        for i in range(4):
            w444_row = k * 4 + i
            w_global_row = (4 + k) * 4 + i
            for r in range(49):
                W[w_global_row][base_r + r] = W444[w444_row][r]
                
    # Block 3: 4x4x1
    # Rank 4
    # U441, V441, W441
    # U441 (16x4), V441 (4x4), W441 (4x4)
    # U441[i*4+j, r] = 1 if r==i?
    # From earlier derivation:
    # U_{i, j, r} = 1 if r==i.
    # V_{j, r} = 1.
    # W_{i', r} = 1 if r==i'.
    
    base_r = 98
    for i in range(4):
        for j in range(4):
            u_global_row = i * 4 + j
            U[u_global_row][base_r + i] = 1
            
    for j in range(4):
        for r in range(4):
            v_global_row = j * 9 + 8 # B3 is col 8
            V[v_global_row][base_r + r] = 1
            
    for i in range(4):
        for r in range(4):
            w_global_row = 8 * 4 + i # C3 is col 8
            W[w_global_row][base_r + r] = 1 if r == i else 0
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
