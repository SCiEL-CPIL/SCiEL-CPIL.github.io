import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    R1 = 49
    R2 = 32
    R = R1 + R2
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # --- 4x4x4 Strassen ---
    
    # 2x2x2 Strassen factors
    u2 = np.array([
        [1, 1, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, -1],
        [1, 1, 1, 1, 0, 0, -1]
    ]) # Wait, previous manual derivation was better.
    # Let's reuse the manual one.
    
    u2 = np.zeros((4, 7))
    v2 = np.zeros((4, 7))
    w2 = np.zeros((4, 7))
    
    # m1
    u2[0,0]=1; u2[3,0]=1
    v2[0,0]=1; v2[3,0]=1
    w2[0,0]=1; w2[3,0]=1
    # m2
    u2[2,1]=1; u2[3,1]=1
    v2[0,1]=1
    w2[2,1]=1; w2[3,1]=-1
    # m3
    u2[0,2]=1
    v2[1,2]=1; v2[3,2]=-1
    w2[0,2]=1; w2[1,2]=1
    # m4
    u2[3,3]=1
    v2[2,3]=1; v2[0,3]=-1
    w2[0,3]=1; w2[1,3]=1
    # m5
    u2[0,4]=1; u2[1,4]=1
    v2[3,4]=1
    w2[1,4]=1; w2[2,4]=-1
    # m6
    u2[2,5]=1; u2[0,5]=-1
    v2[0,5]=1; v2[1,5]=1
    w2[2,5]=1; w2[3,5]=1
    # m7
    u2[1,6]=1; u2[3,6]=-1
    v2[2,6]=1; v2[3,6]=1
    w2[0,6]=1; w2[3,6]=1
    
    # Block indices for 4x4
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    def get_block_indices(block):
        # block is (r, c) where r,c in 0..1
        base = (block[0] * 2) * 4 + (block[1] * 2)
        return [base, base+1, base+4, base+5]
    
    # Strassen Steps for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    steps = [
        # (A_blocks, signs), (B_blocks, signs), (C_blocks, signs)
        # A: [(0,0, 1), (1,1, 1)]
        # B: [(0,0, 1), (1,1, 1)]
        # C: [(0,0, 1), (1,1, 1)]
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,0), 1), ((0,1), 1)]),
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,1), 1), ((1,0), -1)]),
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,0), 1), ((1,1), 1)]),
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]
    
    col = 0
    for step in steps:
        a_blocks, b_blocks, c_blocks = step
        
        # For each of the 7 terms in 2x2x2
        for t in range(7):
            # U
            # u2 is 2x2 A indices
            for i_local in range(4):
                if u2[i_local, t] != 0:
                    val = u2[i_local, t]
                    i_block = i_local // 2
                    i_offset = i_local % 2
                    for a_idx, sign in a_blocks:
                        global_idx = get_block_indices(a_idx)
                        # Map i_local to global
                        # i_local 0,1 -> row 0,1 of block
                        # i_local 2,3 -> row 2,3 of block
                        g_i = global_idx[i_local]
                        U[g_i, col] += sign * val
            
            # V
            for j_local in range(4):
                if v2[j_local, t] != 0:
                    val = v2[j_local, t]
                    for b_idx, sign in b_blocks:
                        global_idx = get_block_indices(b_idx)
                        g_j = global_idx[j_local]
                        # V index is j * p + k
                        # Here p for this subproblem is 4? No, B is 4x4.
                        # V is (m*p, R) = (16, R) for 4x4x4 part.
                        # Wait, V index in final output is j * 6 + k.
                        # For 4x4x4, k is 0..3.
                        # j_local is 0..3 within the 2x2 block.
                        # g_j is the global j index (0..3).
                        # k_local is 0..3 within the 2x2 block.
                        # g_k is the global k index (0..3).
                        
                        # v2 index j_local corresponds to B row?
                        # In 2x2x2, V is 2x2. Indices 0,1,2,3.
                        # j_local 0,1 are row 0, 2,3 are row 1?
                        # No, standard flattening is row-major.
                        # B is 2x2.
                        # B00=0, B01=1, B10=2, B11=3.
                        # So j_local is the index in 0..3.
                        # g_j is the global j.
                        # k_local is the index in 0..3.
                        # g_k is the global k.
                        
                        # We need k_local too.
                        # v2 is indexed by [j_local, t]?
                        # No, v2 is 4x7.
                        # The rows of v2 correspond to B elements.
                        # So row j_local is B element j_local.
                        # But we also need to know which C element it contributes to?
                        # No, V is just the B factor.
                        # The C factor W handles the mapping.
                        
                        # So for V:
                        # Row index in final V is g_j * 6 + g_k.
                        # But g_k is not determined by j_local alone?
                        # In 2x2x2, B is 2x2.
                        # v2 row j_local corresponds to B_{r,c} where r=j_local//2, c=j_local%2.
                        # This B_{r,c} is part of the block B_block.
                        # So g_j = block_row * 2 + r.
                        # g_k = block_col * 2 + c.
                        
                        r = j_local // 2
                        c = j_local % 2
                        g_j = b_idx[0] * 2 + r
                        g_k = b_idx[1] * 2 + c
                        
                        V[g_j * 6 + g_k, col] += sign * val
            
            # W
            for k_local in range(4):
                if w2[k_local, t] != 0:
                    val = w2[k_local, t]
                    r = k_local // 2
                    c = k_local % 2
                    for c_idx, sign in c_blocks:
                        # C block is (c_row, c_col)
                        g_k = c_idx[1] * 2 + c # k is column of C
                        g_i = c_idx[0] * 2 + r # i is row of C
                        # W index is k * n + i = g_k * 4 + g_i
                        W[g_k * 4 + g_i, col] += sign * val
            
            col += 1
    
    # --- 4x4x2 Schoolbook ---
    # A (4x4), B (4x2), C (4x2)
    # k in 4..5 (global)
    # i in 0..3
    # j in 0..3
    
    for i in range(4):
        for j in range(4):
            for k_rel in range(2):
                k_global = 4 + k_rel
                # U: A[i,j]
                U[i * 4 + j, col] = 1
                # V: B[j, k]
                V[j * 6 + k_global, col] = 1
                # W: C[i, k]
                W[k_global * 4 + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Hybrid decomposition: Rank 49 Strassen for 4x4x4 + Rank 32 Schoolbook for 4x4x2, totaling 81 vs current 84", q: 0.9, delta: [0.03, 0.05]}
