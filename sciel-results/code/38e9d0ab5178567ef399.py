"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x2x11 multiplication into 5 independent 2x2x2 blocks
    # and 1 independent 2x2x1 block.
    # The 2x2x2 blocks use Strassen's algorithm (rank 7).
    # The 2x2x1 block uses the trivial algorithm (rank 4).
    # Total rank = 5 * 7 + 4 = 39.
    
    # Strassen's 7 intermediate products for a 2x2 block:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For each 2x2 block, we define the U, V, W vectors for the 7 terms.
    # A is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> u in 0..3
    # B is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> v in 0..3 (local)
    # C is 2x2: indices (0,0), (0,1), (1,0), (1,1) -> w in 0..3 (local)
    
    # U vectors (coefficients for A elements)
    U_strassen = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [0, 0, 1, -1], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    # V vectors (coefficients for B elements)
    V_strassen = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # W vectors (coefficients for C elements)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00, C11
        [0, 0, 1, -1],  # M2: C10, -C11
        [0, 1, 0, 1],   # M3: C01, C11
        [1, 0, 1, 0],   # M4: C00, C10
        [-1, 1, 0, 0],  # M5: -C00, C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Trivial 4 terms for a 2x2x1 block (B has only 1 column, so it's 2x1)
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = a*b0 + c*b1
    # C1 = b*b0 + d*b1
    # Terms:
    # T1: a*b0 -> C0
    # T2: c*b1 -> C0
    # T3: b*b0 -> C1
    # T4: d*b1 -> C1
    U_trivial = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    V_trivial = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    W_trivial = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ]
    
    # Total rank
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    col = 0
    # Process 5 blocks of 2 columns each
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        for t in range(7):
            # U part: same for all blocks
            for i in range(4):
                U[i][col] = U_strassen[t][i]
            
            # V part: local indices 0..3 map to global indices
            # Local B indices: 0->(0,k0), 1->(0,k1), 2->(1,k0), 3->(1,k1)
            # Global V index = row * p + col
            # Row 0: k0, k1 -> 0*11+k0, 0*11+k1
            # Row 1: k0, k1 -> 1*11+k0, 1*11+k1
            v_indices = [k0, k1, 11 + k0, 11 + k1]
            for i in range(4):
                V[v_indices[i]][col] = V_strassen[t][i]
            
            # W part: local indices 0..3 map to global indices
            # Local C indices: 0->(0,k0), 1->(0,k1), 2->(1,k0), 3->(1,k1)
            # Global W index = col * n + row
            # Col k0: 0->k0*2+0, 2->k0*2+1
            # Col k1: 1->k1*2+0, 3->k1*2+1
            w_indices = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]
            for i in range(4):
                W[w_indices[i]][col] = W_strassen[t][i]
            
            col += 1
    
    # Process the last block (1 column)
    k0 = 10
    for t in range(4):
        # U part
        for i in range(4):
            U[i][col] = U_trivial[t][i]
        
        # V part: local indices 0..1 map to global indices
        # Local B indices: 0->(0,k0), 1->(1,k0)
        # Global V index = row * p + col
        v_indices = [k0, 11 + k0]
        for i in range(2):
            V[v_indices[i]][col] = V_trivial[t][i]
        
        # W part: local indices 0..1 map to global indices
        # Local C indices: 0->(0,k0), 1->(1,k0)
        # Global W index = col * n + row
        w_indices = [2 * k0, 2 * k0 + 1]
        for i in range(2):
            W[w_indices[i]][col] = W_trivial[t][i]
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
