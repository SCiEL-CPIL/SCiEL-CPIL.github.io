"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, c_r1, c_r2, k1, k2):
        # A block: rows a_r1, a_r2; cols a_c1, a_c2
        # B block: rows b_r1, b_r2; cols k1, k2
        # C block: rows c_r1, c_r2; cols k1, k2
        
        u_a = a_r1 * m + a_c1
        u_b = a_r1 * m + a_c2
        u_c = a_r2 * m + a_c1
        u_d = a_r2 * m + a_c2
        
        v_e = b_r1 * p + k1
        v_f = b_r1 * p + k2
        v_g = b_r2 * p + k1
        v_h = b_r2 * p + k2
        
        w_c11 = k1 * n + c_r1
        w_c12 = k2 * n + c_r1
        w_c21 = k1 * n + c_r2
        w_c22 = k2 * n + c_r2
        
        # M1 = (a+d)(e+h)
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2 = (c+d)e
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3 = a(f-h)
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4 = d(g-e)
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5 = (a+b)h
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6 = (c-a)(e+f)
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7 = (b-d)(g+h)
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})

    # Block 1: A[0:2, 0:2] * B[0:2, :]
    for k_start in range(0, 8, 2):
        add_strassen_block(0, 0, 1, 1, 0, 0, 1, 1, 0, 1, k_start, k_start + 1)

    # Block 2: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        add_strassen_block(0, 2, 1, 3, 2, 0, 3, 1, 0, 1, k_start, k_start + 1)

    # Block 3: A[2:3, 0:2] * B[0:2, :]
    for k in range(8):
        add_rank_unit({2 * m + 0: 1}, {0 * p + k: 1}, {k * n + 2: 1})
        add_rank_unit({2 * m + 1: 1}, {1 * p + k: 1}, {k * n + 2: 1})

    # Block 4: A[2:3, 2:4] * B[2:4, :]
    for k in range(8):
        add_rank_unit({2 * m + 2: 1}, {2 * p + k: 1}, {k * n + 2: 1})
        add_rank_unit({2 * m + 3: 1}, {3 * p + k: 1}, {k * n + 2: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
