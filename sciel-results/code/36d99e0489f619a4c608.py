"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Part 1: A_R B_R (2x1 x 1x6) -> Rank 12
    # A_R is col 4 of A (rows 0,1). B_R is row 4 of B (cols 0-5).
    # C[i, k] = A[i, 4] * B[4, k]
    for i in range(2):
        for k in range(6):
            u_idx = i * 5 + 4
            v_idx = 4 * 6 + k
            w_idx = k * 2 + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Part 2: A_L B_L (2x4 x 4x6)
    # Split into A_L1 B_L1 (2x2 x 2x6) and A_L2 B_L2 (2x2 x 2x6)
    
    # Sub-part 2a: A_L1 B_L1
    # A_L1: rows 0-1, cols 0-1
    # B_L1: rows 0-1, cols 0-5
    # Split B_L1 into 3 blocks of 2 cols: (0,1), (2,3), (4,5)
    a_i0, a_i1 = 0, 1
    a_j0, a_j1 = 0, 1
    b_i0, b_i1 = 0, 1
    
    # Strassen coefficients
    # M1: (a+d)(x+w) -> C00+1, C11-1
    # M2: (c+d)x -> C10+1, C11+1
    # M3: a(y-w) -> C01+1, C11+1
    # M4: d(z-x) -> C00+1, C10+1
    # M5: (a+b)w -> C00-1, C01+1
    # M6: (c-a)(x+y) -> C11+1
    # M7: (b-d)(z+w) -> C00+1, C11+1
    
    for block_k in range(3):
        k0 = block_k * 2
        k1 = block_k * 2 + 1
        
        # U indices for A_L1
        u_a = a_i0 * 5 + a_j0 # 0
        u_b = a_i0 * 5 + a_j1 # 1
        u_c = a_i1 * 5 + a_j0 # 5
        u_d = a_i1 * 5 + a_j1 # 6
        
        # V indices for B_L1 block
        v_x = b_i0 * 6 + k0
        v_y = b_i0 * 6 + k1
        v_z = b_i1 * 6 + k0
        v_w = b_i1 * 6 + k1
        
        # W indices for C block
        w_c00 = k0 * 2 + a_i0
        w_c01 = k1 * 2 + a_i0
        w_c10 = k0 * 2 + a_i1
        w_c11 = k1 * 2 + a_i1
        
        # M1
        add_term({u_a: 1, u_d: 1}, {v_x: 1, v_w: 1}, {w_c00: 1, w_c11: -1})
        # M2
        add_term({u_c: 1, u_d: 1}, {v_x: 1}, {w_c10: 1, w_c11: 1})
        # M3
        add_term({u_a: 1}, {v_y: 1, v_w: -1}, {w_c01: 1, w_c11: 1})
        # M4
        add_term({u_d: 1}, {v_z: 1, v_x: -1}, {w_c00: 1, w_c10: 1})
        # M5
        add_term({u_a: 1, u_b: 1}, {v_w: 1}, {w_c00: -1, w_c01: 1})
        # M6
        add_term({u_c: 1, u_a: -1}, {v_x: 1, v_y: 1}, {w_c11: 1})
        # M7
        add_term({u_b: 1, u_d: -1}, {v_z: 1, v_w: 1}, {w_c00: 1, w_c11: 1})

    # Sub-part 2b: A_L2 B_L2
    # A_L2: rows 0-1, cols 2-3
    # B_L2: rows 2-3, cols 0-5
    a_i0, a_i1 = 0, 1
    a_j0, a_j1 = 2, 3
    b_i0, b_i1 = 2, 3
    
    for block_k in range(3):
        k0 = block_k * 2
        k1 = block_k * 2 + 1
        
        # U indices for A_L2
        u_a = a_i0 * 5 + a_j0 # 2
        u_b = a_i0 * 5 + a_j1 # 3
        u_c = a_i1 * 5 + a_j0 # 7
        u_d = a_i1 * 5 + a_j1 # 8
        
        # V indices for B_L2 block
        v_x = b_i0 * 6 + k0
        v_y = b_i0 * 6 + k1
        v_z = b_i1 * 6 + k0
        v_w = b_i1 * 6 + k1
        
        # W indices for C block
        w_c00 = k0 * 2 + a_i0
        w_c01 = k1 * 2 + a_i0
        w_c10 = k0 * 2 + a_i1
        w_c11 = k1 * 2 + a_i1
        
        # M1
        add_term({u_a: 1, u_d: 1}, {v_x: 1, v_w: 1}, {w_c00: 1, w_c11: -1})
        # M2
        add_term({u_c: 1, u_d: 1}, {v_x: 1}, {w_c10: 1, w_c11: 1})
        # M3
        add_term({u_a: 1}, {v_y: 1, v_w: -1}, {w_c01: 1, w_c11: 1})
        # M4
        add_term({u_d: 1}, {v_z: 1, v_x: -1}, {w_c00: 1, w_c10: 1})
        # M5
        add_term({u_a: 1, u_b: 1}, {v_w: 1}, {w_c00: -1, w_c01: 1})
        # M6
        add_term({u_c: 1, u_a: -1}, {v_x: 1, v_y: 1}, {w_c11: 1})
        # M7
        add_term({u_b: 1, u_d: -1}, {v_z: 1, v_w: 1}, {w_c00: 1, w_c11: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
