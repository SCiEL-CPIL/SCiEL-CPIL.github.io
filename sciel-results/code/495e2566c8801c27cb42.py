"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(U_dict, V_dict, W_dict, t_idx):
        for (i, j), val in U_dict.items():
            if val != 0:
                U[i * m + j][t_idx] += val
        for (j, k), val in V_dict.items():
            if val != 0:
                V[j * p + k][t_idx] += val
        for (k, i), val in W_dict.items():
            if val != 0:
                W[k * n + i][t_idx] += val

    # Base 2x2x2 terms
    base_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Block offsets
    a_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    b_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    # 7 products
    a_blocks = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    b_blocks = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # Contribution of each P_k to global C blocks
    # C00: P1 + P4 - P5 + P7
    # C01: P3 + P5
    # C10: P2 + P4
    # C11: P1 - P2 + P3 + P6
    # Map: k -> list of (global_block_id, coeff)
    # Global block ids: 0:00, 1:01, 2:10, 3:11
    contrib = [
        [(0, 1), (3, 1)], # P1 -> C00, C11
        [(2, 1), (3, -1)], # P2 -> C10, C11
        [(1, 1), (3, 1)], # P3 -> C01, C11
        [(0, 1), (2, 1)], # P4 -> C00, C10
        [(0, -1), (1, 1)], # P5 -> C00, C01
        [(3, 1)], # P6 -> C11
        [(0, 1)] # P7 -> C00
    ]
    
    t_idx = 0
    for k in range(7):
        a_b = a_blocks[k]
        b_b = b_blocks[k]
        for s in range(7):
            U_base, V_base, W_base = base_terms[s]
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (bi, bj), coeff_a in a_b.items():
                i_off, j_off = a_offsets[bi]
                for (li, lj), val_u in U_base.items():
                    i = i_off + li
                    j = j_off + lj
                    U_global[(i, j)] = U_global.get((i, j), 0) + coeff_a * val_u
                    
            for (bi, bj), coeff_b in b_b.items():
                i_off, j_off = b_offsets[bi]
                for (li, lj), val_v in V_base.items():
                    j = i_off + li
                    k_col = j_off + lj
                    V_global[(j, k_col)] = V_global.get((j, k_col), 0) + coeff_b * val_v
                    
            # W_global: The result of the 2x2 product is a 2x2 matrix.
            # We need to add it to the global C blocks specified in contrib[k].
            for (gb_id, coeff_w) in contrib[k]:
                i_off, j_off = a_offsets[gb_id] # Wait, C blocks use same offsets as A?
                # C00 is C[0:2, 0:2]. i in 0,1; k in 0,1.
                # C01 is C[0:2, 2:4]. i in 0,1; k in 2,3.
                # C10 is C[2:4, 0:2]. i in 2,3; k in 0,1.
                # C11 is C[2:4, 2:4]. i in 2,3; k in 2,3.
                # So offsets for C are:
                c_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
                i_off, k_off = c_offsets[gb_id]
                for (li, lk), val_w in W_base.items():
                    i = i_off + li
                    k = k_off + lk
                    W_global[(k, i)] = W_global.get((k, i), 0) + coeff_w * val_w
            
            add_term(U_global, V_global, W_global, t_idx)
            t_idx += 1
            
    # Add 16 terms for 4x4x1 (5th column)
    for i in range(4):
        for j in range(4):
            U_dict = {(i, j): 1}
            V_dict = {(j, 4): 1}
            W_dict = {(4, i): 1}
            add_term(U_dict, V_dict, W_dict, t_idx)
            t_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
