"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen parameters
    # s: 0 to 6
    # A_blocks: dict of block_name -> coeff
    # B_blocks: dict of block_name -> coeff
    # W_coeffs: dict of out_block -> coeff
    
    strassen_params = [
        # M1
        {'A': {'11': 1, '22': 1}, 'B': {'11': 1, '22': 1}, 'W': {'11': 1, '22': 1}},
        # M2
        {'A': {'21': 1, '22': 1}, 'B': {'11': 1}, 'W': {'21': 1, '22': -1}},
        # M3
        {'A': {'11': 1}, 'B': {'12': 1, '22': -1}, 'W': {'12': 1, '22': 1}},
        # M4
        {'A': {'22': 1}, 'B': {'21': 1, '11': -1}, 'W': {'11': 1, '21': 1}},
        # M5
        {'A': {'11': 1, '12': 1}, 'B': {'22': 1}, 'W': {'11': -1, '12': 1}},
        # M6
        {'A': {'21': 1, '11': -1}, 'B': {'11': 1, '12': 1}, 'W': {'22': 1}},
        # M7
        {'A': {'12': 1, '22': -1}, 'B': {'21': 1, '22': 1}, 'W': {'11': 1}},
    ]
    
    # Block mappings
    # A blocks: (i, j_offset)
    # A11: i=0, j=0,1 -> A[0, q]
    # A12: i=0, j=2,3 -> A[0, q+2]
    # A21: i=1, j=0,1 -> A[1, q]
    # A22: i=1, j=2,3 -> A[1, q+2]
    A_map = {
        '11': (0, 0),
        '12': (0, 2),
        '21': (1, 0),
        '22': (1, 2)
    }
    
    # B blocks: (j_offset, k_offset)
    # B11: j=0,1; k=0..3 -> B[q, l]
    # B12: j=0,1; k=4..7 -> B[q, l+4]
    # B21: j=2,3; k=0..3 -> B[q+2, l]
    # B22: j=2,3; k=4..7 -> B[q+2, l+4]
    B_map = {
        '11': (0, 0),
        '12': (0, 4),
        '21': (2, 0),
        '22': (2, 4)
    }
    
    # W blocks: (i, k_offset)
    # 11: i=0, k=0..3 -> W[l*2 + 0]
    # 12: i=0, k=4..7 -> W[(l+4)*2 + 0]
    # 21: i=1, k=0..3 -> W[l*2 + 1]
    # 22: i=1, k=4..7 -> W[(l+4)*2 + 1]
    W_map = {
        '11': (0, 0),
        '12': (0, 4),
        '21': (1, 0),
        '22': (1, 4)
    }
    
    for s, params in enumerate(strassen_params):
        for q in range(2):
            for l in range(4):
                r = s * 8 + q * 4 + l
                
                # Update U
                for block, coeff in params['A'].items():
                    i, j_off = A_map[block]
                    j = j_off + q
                    u_idx = i * m + j
                    U[u_idx][r] += coeff
                
                # Update V
                for block, coeff in params['B'].items():
                    j_off, k_off = B_map[block]
                    j = j_off + q
                    k = k_off + l
                    v_idx = j * p + k
                    V[v_idx][r] += coeff
                
                # Update W
                for block, coeff in params['W'].items():
                    i, k_off = W_map[block]
                    k = k_off + l
                    w_idx = k * n + i
                    W[w_idx][r] += coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
