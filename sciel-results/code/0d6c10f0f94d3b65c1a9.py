"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2
    # u: (i, j), v: (j, k), w: (k, i)
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # The 7 Strassen products:
    # 1: (a00+a11)(b00+b11) -> c00, c11
    # 2: (a10+a11)b00 -> c10, c11
    # 3: a00(b01-b11) -> c01, c11
    # 4: a11(b10-b00) -> c00, c10
    # 5: (a00+a01)b11 -> c01, c00
    # 6: (a10-a00)(b00+b01) -> c11
    # 7: (a01-a11)(b10+b11) -> c00
    
    # We define the vectors for one 2x2x2 block.
    # Then we instantiate it twice: once for k in {0,1} and once for k in {2,3}.
    
    strassen_terms = [
        # (u_entries, v_entries, w_entries)
        # u_entries: list of (i, j, val)
        # v_entries: list of (j, k, val)
        # w_entries: list of (k, i, val)
        
        # Term 1
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        # Term 2
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)]
        ),
        # Term 3
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)]
        ),
        # Term 4
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)]
        ),
        # Term 5
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(1,0,1), (0,0,-1)]
        ),
        # Term 6
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        # Term 7
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    # Fill the matrices for the two blocks
    # Block 0: k_offset = 0, col_offset = 0
    # Block 1: k_offset = 2, col_offset = 7
    
    for block_idx, k_offset in enumerate([0, 2]):
        col_base = block_idx * 7
        
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(strassen_terms):
            col = col_base + t_idx
            
            # Fill U
            for i, j, val in u_entries:
                u_idx = i * m + j
                U[u_idx][col] += val
            
            # Fill V
            for j, k, val in v_entries:
                k_local = k + k_offset
                v_idx = j * p + k_local
                V[v_idx][col] += val
            
            # Fill W
            for k, i, val in w_entries:
                k_local = k + k_offset
                w_idx = k_local * n + i
                W[w_idx][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
