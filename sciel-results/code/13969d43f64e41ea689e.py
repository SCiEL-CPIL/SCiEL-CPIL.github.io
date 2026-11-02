"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(i, j):
        return i * p + j
        
    def w_idx(i, j):
        return j * n + i

    # Outer Strassen terms definitions
    # Block IDs: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    outer_terms = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)]}
    ]
    
    c_contribs = [
        [(0, 1), (3, 1)],      # M1
        [(2, 1), (3, -1)],     # M2
        [(1, 1), (3, 1)],      # M3
        [(0, 1), (2, 1)],      # M4
        [(0, -1), (1, 1)],     # M5
        [(3, 1)],              # M6
        [(0, 1)]               # M7
    ]

    inner_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]

    def get_block_elements(block_id):
        br, bc = divmod(block_id, 2)
        base_r = br * 2
        base_c = bc * 2
        return [
            (base_r + 0, base_c + 0),
            (base_r + 0, base_c + 1),
            (base_r + 1, base_c + 0),
            (base_r + 1, base_c + 1)
        ]
    
    z_contribs = [
        [(0,0, 1), (1,1, 1)],
        [(1,0, 1), (1,1, -1)],
        [(0,1, 1), (1,1, 1)],
        [(0,0, 1), (1,0, 1)],
        [(0,0, -1), (0,1, 1)],
        [(1,1, 1)],
        [(0,0, 1)]
    ]

    for t_idx, term in enumerate(outer_terms):
        A_blocks = term['A']
        B_blocks = term['B']
        C_blocks = c_contribs[t_idx]
        
        for s_idx, (X_coeffs, Y_coeffs) in enumerate(inner_products):
            rank_idx = t_idx * 7 + s_idx
            
            U_contrib = {}
            V_contrib = {}
            
            for (xr, xc), x_coeff in X_coeffs.items():
                for b_id, b_coeff in A_blocks:
                    elems = get_block_elements(b_id)
                    global_i, global_j = elems[xr * 2 + xc]
                    total_coeff = x_coeff * b_coeff
                    key = (global_i, global_j)
                    U_contrib[key] = U_contrib.get(key, 0) + total_coeff

            for (yr, yc), y_coeff in Y_coeffs.items():
                for b_id, b_coeff in B_blocks:
                    elems = get_block_elements(b_id)
                    global_i, global_j = elems[yr * 2 + yc]
                    total_coeff = y_coeff * b_coeff
                    key = (global_i, global_j)
                    V_contrib[key] = V_contrib.get(key, 0) + total_coeff

            for (i, j), coeff in U_contrib.items():
                if coeff != 0:
                    U[u_idx(i, j)][rank_idx] = coeff
                    
            for (i, j), coeff in V_contrib.items():
                if coeff != 0:
                    V[v_idx(i, j)][rank_idx] = coeff

            for (zr, zc, z_coeff) in z_contribs[s_idx]:
                for c_id, c_coeff in C_blocks:
                    elems = get_block_elements(c_id)
                    global_i, global_j = elems[zr * 2 + zc]
                    total_coeff = z_coeff * c_coeff
                    w_i = w_idx(global_i, global_j)
                    W[w_i][rank_idx] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
