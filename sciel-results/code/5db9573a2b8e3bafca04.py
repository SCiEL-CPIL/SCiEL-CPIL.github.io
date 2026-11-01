import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Rank 56 solution
    # A is 2x4, split into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
    # B is 4x8, split into 4 column groups of 2.
    # For each group c (0..3), B has two 2x2 blocks: B1c (rows 0-1), B2c (rows 2-3)
    # C_c (2x2) = A1 B1c + A2 B2c
    # Each term A_i B_jc is a 2x2x2 multiplication (rank 7)
    # Total 4 groups * 2 terms * 7 rank = 56
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 rank-7 vectors
    # U, V, W for a single 2x2x2 multiplication
    # Indices for U: 0..3 (i,j), V: 0..3 (j,k), W: 0..3 (k,i)
    # We need to map these to the global indices.
    
    # Global U index: i * 4 + j. i in 0..1, j in 0..3.
    # Global V index: j * 8 + k. j in 0..3, k in 0..7.
    # Global W index: k * 2 + i. k in 0..7, i in 0..1.
    
    # Strassen terms for 2x2 matrix multiplication
    # Local indices for 2x2:
    # u_idx: i*2+j
    # v_idx: j*2+k
    # w_idx: k*2+i
    
    strassen_terms = [
        # (u_mask, v_mask, w_mask)
        # 1: (A00+A11)(B00+B11)
        { 'u': [0, 3], 'v': [0, 3], 'w': [0, 1, 2, 3] }, # Wait, W is output.
        # Let's use the standard 7 vectors.
    ]
    
    # Standard Strassen 7 products for 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We define the 7 rank-1 decompositions for the 2x2x2 tensor.
    # Each term r has U_r, V_r, W_r vectors of length 4.
    
    # Term 0: P1
    # U: A00, A11 -> indices 0, 3. Values 1, 1.
    # V: B00, B11 -> indices 0, 3. Values 1, 1.
    # W: C00, C01, C10, C11?
    # P1 contributes to C00, C01, C10, C11 with +1.
    # W indices: 0 (k=0,i=0), 1 (k=1,i=0), 2 (k=0,i=1), 3 (k=1,i=1).
    # So W is [1, 1, 1, 1].
    
    def make_term(u_vals, v_vals, w_vals):
        # u_vals, v_vals, w_vals are dicts {local_idx: val}
        return u_vals, v_vals, w_vals

    # Local indices:
    # U: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # V: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # W: 0(0,0), 1(1,0), 2(0,1), 3(1,1)  (k*n+i)
    
    s_terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        # P2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}), # C10, C11
        # P3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {0: 1, 1: 1}), # C00, C01
        # P4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 3: 1}), # C10, C11
        # P5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}), # C00, C01, C10, C11?
        # Wait, P5 = (A00+A01)B11.
        # C00 += A00 B11? No.
        # Strassen:
        # C00 = P1 - P4 + P3
        # C01 = P5 + P3
        # C10 = P2 + P4
        # C11 = P1 - P2 + P5
        # Let's re-verify the W vectors based on this.
    ]
    
    # Correct Strassen W vectors:
    # P1: C00(+), C11(+) -> W: 0:+1, 3:+1. (C00 is idx 0, C11 is idx 3)
    # P2: C10(+), C11(-) -> W: 2:+1, 3:-1.
    # P3: C00(+), C01(+) -> W: 0:+1, 1:+1.
    # P4: C00(-), C10(+) -> W: 0:-1, 2:+1.
    # P5: C01(+), C11(+) -> W: 1:+1, 3:+1.
    # P6: C10(+) -> W: 2:+1. (P6 = (A10-A00)(B00+B01) -> C10 += P6)
    # P7: C01(+) -> W: 1:+1. (P7 = (A01-A11)(B10+B11) -> C01 += P7)
    
    # Let's check P6 and P7 U/V.
    # P6: (A10-A00)(B00+B01)
    # U: A10(1), A00(-1) -> {2: 1, 0: -1}
    # V: B00(1), B01(1) -> {0: 1, 1: 1}
    # W: C10(1) -> {2: 1}
    
    # P7: (A01-A11)(B10+B11)
    # U: A01(1), A11(-1) -> {1: 1, 3: -1}
    # V: B10(1), B11(1) -> {2: 1, 3: 1}
    # W: C01(1) -> {1: 1}
    
    strassen_defs = [
        # (U, V, W)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # P2
        ({0: 1}, {1: 1, 3: -1}, {0: 1, 1: 1}),       # P3
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),      # P4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),        # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1}),       # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1}),       # P7
    ]
    
    col = 0
    # 4 column groups c = 0, 1, 2, 3
    # Each group has 2 terms: A1*B1c and A2*B2c
    for c in range(4):
        # Term 1: A1 * B1c
        # A1 is rows 0-1, cols 0-1 of A.
        # B1c is rows 0-1, cols 2c..2c+1 of B.
        # Output C_c is rows 0-1, cols 2c..2c+1 of C.
        
        for s_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_defs):
            for u_l, u_v in u_loc.items():
                # u_l is local 0..3 for 2x2.
                # i = u_l // 2, j = u_l % 2.
                i = u_l // 2
                j = u_l % 2
                # Global U index: i * 4 + j.
                # But j is local 0,1. Global j for A1 is 0,1.
                # So global j = j.
                u_g_idx = i * 4 + j
                U[u_g_idx, col] += u_v
            
            for v_l, v_v in v_loc.items():
                # v_l is local 0..3 for 2x2.
                # j = v_l // 2, k = v_l % 2.
                j = v_l // 2
                k = v_l % 2
                # Global V index: j * 8 + k.
                # j is local 0,1. Global j for B1c is 0,1.
                # k is local 0,1. Global k for col group c is 2c, 2c+1.
                # So global k = 2*c + k.
                v_g_idx = j * 8 + (2*c + k)
                V[v_g_idx, col] += v_v
            
            for w_l, w_v in w_loc.items():
                # w_l is local 0..3 for 2x2.
                # k = w_l // 2, i = w_l % 2.
                k = w_l // 2
                i = w_l % 2
                # Global W index: k * 2 + i.
                # k is local 0,1. Global k is 2c, 2c+1.
                # i is local 0,1. Global i is 0,1.
                # So global k = 2*c + k.
                w_g_idx = (2*c + k) * 2 + i
                W[w_g_idx, col] += w_v
            
            col += 1
        
        # Term 2: A2 * B2c
        # A2 is rows 0-1, cols 2-3 of A.
        # B2c is rows 2-3, cols 2c..2c+1 of B.
        
        for s_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_defs):
            for u_l, u_v in u_loc.items():
                i = u_l // 2
                j = u_l % 2
                # Global j for A2 is 2,3.
                # So global j = 2 + j.
                u_g_idx = i * 4 + (2 + j)
                U[u_g_idx, col] += u_v
            
            for v_l, v_v in v_loc.items():
                j = v_l // 2
                k = v_l % 2
                # Global j for B2c is 2,3.
                # So global j = 2 + j.
                # Global k is 2c + k.
                v_g_idx = (2 + j) * 8 + (2*c + k)
                V[v_g_idx, col] += v_v
            
            for w_l, w_v in w_loc.items():
                k = w_l // 2
                i = w_l % 2
                # Global k is 2c + k.
                # Global i is i.
                w_g_idx = (2*c + k) * 2 + i
                W[w_g_idx, col] += w_v
            
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
