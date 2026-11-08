import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-98 decomposition for 4x4x8 matrix multiplication.
    Uses recursive Strassen to get a rank-49 decomposition for 4x4x4,
    then duplicates it for the two 4x4x4 halves of the 4x4x8 multiplication.
    """
    n, m, p = DIMENSIONS
    
    # 1. Get base 2x2x2 Strassen decomposition (Rank 7)
    # Returns U, V, W of shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    def get_2x2_decomp():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1: (a+d)(e+h) -> C00, C11
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1 # C00 (0,0), C11 (1,1) -> W[0], W[3]
        
        # M2: (c+d)e -> C10
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[1][1] = 1 # C10 (0,1) -> W[1]
        
        # M3: a(f-h) -> C01
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[2][2] = 1 # C01 (1,0) -> W[2]
        
        # M4: d(g-e) -> C00, C10
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[1][3] = 1 # C00 (0,0), C10 (0,1)
        
        # M5: (a+b)h -> C00, C01
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[2][4] = 1 # C00 (0,0), C01 (1,0)
        
        # M6: (c-a)(e+f) -> C11
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1 # C11 (1,1)
        
        # M7: (b-d)(g+h) -> C00
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1 # C00 (0,0)
        
        return u, v, w

    u2, v2, w2 = get_2x2_decomp()

    # 2. Build 4x4x4 decomposition (Rank 49)
    def get_4x4_decomp():
        u4 = [[0]*49 for _ in range(16)]
        v4 = [[0]*49 for _ in range(16)]
        w4 = [[0]*49 for _ in range(16)]
        
        # Strassen terms at 4x4 level
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Define terms:
        # term_idx: 0..6
        # a_coeffs: dict block_id -> coeff
        # b_coeffs: dict block_id -> coeff
        # c_coeffs: dict block_id -> coeff (for W)
        
        terms = [
            # M1
            {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}}, # C00, C11
            # M2
            {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1}}, # C10
            # M3
            {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1}}, # C01
            # M4
            {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}}, # C00, C10
            # M5
            {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}}, # C00, C01
            # M6
            {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}}, # C11
            # M7
            {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}}, # C00
        ]
        
        def get_block_offsets(block_id):
            # block_id: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
            r = (block_id // 2) * 2
            c = (block_id % 2) * 2
            return r, c
        
        for t_idx, term in enumerate(terms):
            col_base = t_idx * 7
            
            for c_inner in range(7):
                col = col_base + c_inner
                
                # Accumulate U
                for a_block, coeff in term['a'].items():
                    r_off, c_off = get_block_offsets(a_block)
                    for li in range(2):
                        for lj in range(2):
                            u_idx = (r_off + li) * 4 + (c_off + lj)
                            u4[u_idx][col] += coeff * u2[li*2+lj][c_inner]
                
                # Accumulate V
                for b_block, coeff in term['b'].items():
                    r_off, c_off = get_block_offsets(b_block)
                    for li in range(2):
                        for lj in range(2):
                            v_idx = (r_off + li) * 4 + (c_off + lj)
                            v4[v_idx][col] += coeff * v2[li*2+lj][c_inner]
                            
                # Accumulate W
                for c_block, coeff in term['c'].items():
                    r_off, c_off = get_block_offsets(c_block)
                    for li in range(2):
                        for lj in range(2):
                            # W index is k*4 + i.
                            # C block (r_off, c_off) corresponds to rows r_off..r_off+1, cols c_off..c_off+1.
                            # Global row k = r_off + li.
                            # Global col i = c_off + lj.
                            # W index = i * 4 + k.
                            w_idx = (c_off + lj) * 4 + (r_off + li)
                            w4[w_idx][col] += coeff * w2[li*2+lj][c_inner]
                            
        return u4, v4, w4

    u4, v4, w4 = get_4x4_decomp()
    
    # 3. Build 4x4x8 decomposition (Rank 98)
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First half (k=0..3)
    for i in range(16):
        for c in range(49):
            U[i][c] = u4[i][c]
            
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + k
            u4_v_idx = j * 4 + k
            for c in range(49):
                V[v_idx][c] = v4[u4_v_idx][c]
                
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            u4_w_idx = k * 4 + i
            for c in range(49):
                W[w_idx][c] = w4[u4_w_idx][c]
                
    # Second half (k=4..7)
    for i in range(16):
        for c in range(49):
            U[i][49+c] = u4[i][c]
            
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + (4+k)
            u4_v_idx = j * 4 + k
            for c in range(49):
                V[v_idx][49+c] = v4[u4_v_idx][c]
                
    for k in range(4):
        for i in range(4):
            w_idx = (4+k) * 4 + i
            u4_w_idx = k * 4 + i
            for c in range(49):
                W[w_idx][49+c] = w4[u4_w_idx][c]
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
