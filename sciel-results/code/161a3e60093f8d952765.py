def construct():
    n, m, p = 4, 4, 9
    rank_444 = 49
    rank_441 = 4
    total_rank = 2 * rank_444 + rank_441 # 102
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x 102
    V = [[0] * total_rank for _ in range(m * p)] # 36 x 102
    W = [[0] * total_rank for _ in range(p * n)] # 36 x 102
    
    # Helper to build 4x4x4 Strassen (Rank 49)
    def build_4x4x4_strassen(offset_u, offset_v, offset_w, rank_offset):
        # Get base 2x2 Strassen
        U2, V2, W2 = get_2x2_strassen()
        
        # Coefficients for the 7 meta-multiplications
        # c[k][b] is coeff of A_block_b in M_k
        # d[k][b] is coeff of B_block_b in M_k
        # w[k][b] is coeff of C_block_b in reconstruction from M_k
        
        c = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        d = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, 0, 1, -1],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        w = [
            [1, 0, 0, -1], # C00, C01, C10, C11 coeffs for M1
            [0, 0, -1, 1], # M2
            [-1, 1, 0, 0], # M3 -- Wait, check reconstruction again
            [1, 0, -1, 0], # M4
            [0, -1, 0, 1], # M5
            [0, 0, 0, 1],  # M6
            [1, 0, 0, 0]   # M7
        ]
        
        # Re-verify reconstruction coefficients
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's build w matrix properly: w[k][block_idx]
        # block_idx: 0=C00, 1=C01, 2=C10, 3=C11
        w = [
            [1, 0, 0, 1],  # M1: C00+=1, C11+=1
            [0, 0, 1, -1], # M2: C10+=1, C11-=1
            [0, 1, 0, 1],  # M3: C01+=1, C11+=1
            [1, 0, 1, 0],  # M4: C00+=1, C10+=1
            [-1, 1, 0, 0], # M5: C00-=1, C01+=1
            [0, 0, 0, 1],  # M6: C11+=1
            [1, 0, 0, 0]   # M7: C00+=1
        ]
        
        # Check C00: 1(M1) - 1(M5) + 1(M7) = 1. Correct.
        # Check C01: 1(M3) + 1(M5) = 1. Correct.
        # Check C10: 1(M2) + 1(M4) = 1. Correct.
        # Check C11: 1(M1) - 1(M2) + 1(M3) + 1(M6) = 1. Correct.
        
        # Now fill U, V, W
        for k in range(7):
            for b_a in range(4):
                coeff_a = c[k][b_a]
                if coeff_a != 0:
                    for r in range(4):
                        # Map block b_a and local r to global index in A
                        gr = 2 * (b_a // 2) + (r // 2)
                        gc = 2 * (b_a % 2) + (r % 2)
                        idx_a = 4 * gr + gc
                        s = r # s is the local column in 2x2? No, U2 is 4x7.
                        # U2[r][s] where r is row, s is mult index.
                        # We need to iterate over the 7 mults of the base Strassen.
                        for s in range(7):
                            if U2[r][s] != 0:
                                U[idx_a][rank_offset + 7*k + s] += coeff_a * U2[r][s]
            
            for b_b in range(4):
                coeff_b = d[k][b_b]
                if coeff_b != 0:
                    for r in range(4):
                        gr = 2 * (b_b // 2) + (r // 2)
                        gc = 2 * (b_b % 2) + (r % 2)
                        # V corresponds to B.
                        # In the 4x4x4 subproblem, B is 4x4.
                        # But in the full problem, B is 4x9.
                        # We are filling a specific 4x4 block of B.
                        # The offset_v tells us which 4x4 block of B we are writing to.
                        # offset_v is the starting row index in V for this block.
                        idx_b = offset_v + 4 * gr + gc
                        for s in range(7):
                            if V2[r][s] != 0:
                                V[idx_b][rank_offset + 7*k + s] += coeff_b * V2[r][s]
            
            for b_c in range(4):
                coeff_c = w[k][b_c]
                if coeff_c != 0:
                    for r in range(4):
                        gr = 2 * (b_c // 2) + (r // 2)
                        gc = 2 * (b_c % 2) + (r % 2)
                        # W corresponds to C.
                        # offset_w is the starting row index in W for this block.
                        idx_c = offset_w + 4 * gr + gc
                        for s in range(7):
                            if W2[r][s] != 0:
                                W[idx_c][rank_offset + 7*k + s] += coeff_c * W2[r][s]
    
    # Block 1: Cols 0-3 of B, Cols 0-3 of C
    # A is full 4x4.
    # V offset: 0 (rows 0-3)
    # W offset: 0 (rows 0-3)
    build_4x4x4_strassen(0, 0, 0, 0)
    
    # Block 2: Cols 4-7 of B, Cols 4-7 of C
    # A is full 4x4.
    # V offset: 16 (rows 4-7? No, B is 4x9. Rows 0-3, Cols 4-7.
    # V index = row * 9 + col.
    # For Block 2, cols are 4,5,6,7.
    # So V rows involved are 0..3, but columns 4..7.
    # Wait, V shape is (m*p, rank) = (36, rank).
    # Index in V is $i \cdot 9 + j$.
    # For B block cols 4-7, the indices are $0 \cdot 9+4 \dots 3 \cdot 9+7$.
    # These are not contiguous rows 16-19.
    # My `build_4x4x4_strassen` assumed contiguous rows in V.
    # I need to adjust the mapping.
    
    # Let's rewrite the mapping logic inside the function or pass a mapping.
    # It's easier to just write the loops manually for the two blocks if the offsets are complex.
    # Or, define a mapping function for V and W indices.
    
    # Let's refactor build_4x4x4_strassen to accept index mappings.
    
    def map_a(idx): return idx # A is always 0-15
    def map_v_block1(idx): 
        # idx is 0-15 in local 4x4 block
        r = idx // 4
        c = idx % 4
        # Global col = c
        # Global row = r
        return r * 9 + c
    def map_v_block2(idx):
        r = idx // 4
        c = idx % 4
        # Global col = 4 + c
        # Global row = r
        return r * 9 + (4 + c)
    def map_v_block3(idx):
        r = idx // 4
        c = idx % 4
        # Global col = 8 + c (but c is always 0 in 4x1 case? No, 4x1 is just col 8)
        # For 4x1, we don't use this block builder.
        return r * 9 + 8
        
    def map_w_block1(idx):
        r = idx // 4
        c = idx % 4
        return r * 4 + c # C is 9x4? No, C is 4x9.
        # Wait, W shape is (p*n, rank) = (36, rank).
        # "Output factor follows the official trace convention W[k*n+i,r]"
        # This usually means W is indexed by output element.
        # If C is 4x9, and we flatten it, what is the order?
        # Usually row-major: $i \cdot 9 + j$.
        # But the prompt says "W[k*n+i,r]".
        # If n=4, p=9.
        # Does k correspond to the 9 dimension and i to the 4 dimension?
        # "W[k*n+i]" -> k is 0..8, i is 0..3.
        # This implies column-major flattening of C?
        # Or k is the column index and i is the row index?
        # If C is 4x9, C[i][j].
        # If W index is $k \cdot 4 + i$, then k is the column (0-8) and i is the row (0-3).
        # Let's assume W index = $j \cdot 4 + i$ where $C[i][j]$.
        
        # Let's check the previous failed code.
        # W shape was (36, rank).
        # In Block 3, it used W[32+i].
        # 32 = 8 * 4. So k=8, i=0..3.
        # This matches $j \cdot 4 + i$.
        
    # So for W:
    def map_w_block1(idx):
        r = idx // 4
        c = idx % 4
        # Local r is row, c is col.
        # Global row = r, Global col = c.
        # W index = Global Col * 4 + Global Row.
        return c * 4 + r
        
    def map_w_block2(idx):
        r = idx // 4
        c = idx % 4
        # Global row = r, Global col = 4 + c.
        return (4 + c) * 4 + r

    # Now rebuild with mappings
    
    def build_4x4x4_strassen(map_u, map_v, map_w, rank_offset):
        U2, V2, W2 = get_2x2_strassen()
        c = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        d = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        w = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        
        for k in range(7):
            for b_a in range(4):
                coeff_a = c[k][b_a]
                if coeff_a != 0:
                    for r in range(4):
                        gr = 2 * (b_a // 2) + (r // 2)
                        gc = 2 * (b_a % 2) + (r % 2)
                        idx_a = 4 * gr + gc
                        g_idx_a = map_u(idx_a)
                        for s in range(7):
                            if U2[r][s] != 0:
                                U[g_idx_a][rank_offset + 7*k + s] += coeff_a * U2[r][s]
            
            for b_b in range(4):
                coeff_b = d[k][b_b]
                if coeff_b != 0:
                    for r in range(4):
                        gr = 2 * (b_b // 2) + (r // 2)
                        gc = 2 * (b_b % 2) + (r % 2)
                        idx_b = 4 * gr + gc
                        g_idx_b = map_v(idx_b)
                        for s in range(7):
                            if V2[r][s] != 0:
                                V[g_idx_b][rank_offset + 7*k + s] += coeff_b * V2[r][s]
            
            for b_c in range(4):
                coeff_c = w[k][b_c]
                if coeff_c != 0:
                    for r in range(4):
                        gr = 2 * (b_c // 2) + (r // 2)
                        gc = 2 * (b_c % 2) + (r % 2)
                        idx_c = 4 * gr + gc
                        g_idx_c = map_w(idx_c)
                        for s in range(7):
                            if W2[r][s] != 0:
                                W[g_idx_c][rank_offset + 7*k + s] += coeff_c * W2[r][s]

    # Block 1
    build_4x4x4_strassen(map_a, map_v_block1, map_w_block1, 0)
    
    # Block 2
    build_4x4x4_strassen(map_a, map_v_block2, map_w_block2, 49)
    
    # Block 3: 4x4x1
    # Cols 8 of B, Cols 8 of C.
    # Rank 4.
    # U: A is 4x4.
    # V: B is 4x1 (Col 8).
    # W: C is 4x1 (Col 8).
    for i in range(4):
        for k in range(4):
            m_idx = 98 + i
            # A[i][k]
            U[4*i + k][m_idx] = 1
            # B[k][8]
            V[k*9 + 8][m_idx] = 1
            # C[i][8]
            W[8*4 + i][m_idx] = 1
            
    return {"U": U, "V": V, "W": W}
