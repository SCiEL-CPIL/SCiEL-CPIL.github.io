"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using 7 Strassen blocks of size 2x2x2
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2 multiplication
    # Local indices: A(4), B(4), C(4)
    # u, v, w are lists of length 4
    strassen_terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [-1, 0, 1, 0]), # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 1, -1]), # M4
        ([0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]),   # M5
        ([0, 1, 1, 0], [1, 0, 0, -1], [0, 0, -1, 0]), # M6
        ([1, 1, 0, 0], [0, 1, -1, 0], [-1, 1, 0, 0]), # M7
    ]
    
    # Map local indices to global indices
    # Global A: 0..3
    # Global B: j*14 + k
    # Global W: k*2 + i
    
    col_offset = 0
    for pair_idx in range(p // 2):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # Local B to Global B mapping
        # l=0 -> j=0, k=k0 -> 0*14 + k0
        # l=1 -> j=0, k=k1 -> 0*14 + k1
        # l=2 -> j=1, k=k0 -> 1*14 + k0
        # l=3 -> j=1, k=k1 -> 1*14 + k1
        b_map = [k0, k1, 14 + k0, 14 + k1]
        
        # Local C to Global W mapping
        # l=0 -> i=0, k=k0 -> k0*2 + 0
        # l=1 -> i=0, k=k1 -> k1*2 + 0
        # l=2 -> i=1, k=k0 -> k0*2 + 1
        # l=3 -> i=1, k=k1 -> k1*2 + 1
        w_map = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]
        
        for u, v, w in strassen_terms:
            # Fill U
            for a_idx in range(4):
                if u[a_idx] != 0:
                    U[a_idx][col_offset] = u[a_idx]
            
            # Fill V
            for l_idx in range(4):
                if v[l_idx] != 0:
                    global_b_idx = b_map[l_idx]
                    V[global_b_idx][col_offset] = v[l_idx]
            
            # Fill W
            for l_idx in range(4):
                if w[l_idx] != 0:
                    global_w_idx = w_map[l_idx]
                    W[global_w_idx][col_offset] = w[l_idx]
            
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
