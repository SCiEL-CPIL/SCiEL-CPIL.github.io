"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        # u_indices: list of (global_idx, coeff)
        # v_indices: list of (global_idx, coeff)
        # w_indices: list of (global_idx, coeff)
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff
        col += 1

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        # W[k*n + i, r] -> index is k*n + i
        return c * n + r

    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A block starts at (a_r0, a_c0), size 2x2
        # B block starts at (b_r0, b_c0), size 2x2
        # C block starts at (c_r0, c_c0), size 2x2
        
        # M1
        add_term(
            [(a_idx(a_r0, a_c0), 1), (a_idx(a_r0+1, a_c0+1), 1)],
            [(b_idx(b_r0, b_c0), 1), (b_idx(b_r0+1, b_c0+1), 1)],
            [(c_idx(c_r0, c_c0), 1), (c_idx(c_r0+1, c_c0+1), 1)]
        )
        # M2
        add_term(
            [(a_idx(a_r0+1, a_c0), 1), (a_idx(a_r0+1, a_c0+1), 1)],
            [(b_idx(b_r0, b_c0), 1)],
            [(c_idx(c_r0+1, c_c0), 1)]
        )
        # M3
        add_term(
            [(a_idx(a_r0, a_c0), 1)],
            [(b_idx(b_r0, b_c0+1), 1), (b_idx(b_r0+1, b_c0+1), -1)],
            [(c_idx(c_r0, c_c0+1), 1)]
        )
        # M4
        add_term(
            [(a_idx(a_r0+1, a_c0+1), 1)],
            [(b_idx(b_r0+1, b_c0), 1), (b_idx(b_r0, b_c0), -1)],
            [(c_idx(c_r0, c_c0), 1), (c_idx(c_r0+1, c_c0), 1)]
        )
        # M5
        add_term(
            [(a_idx(a_r0, a_c0), 1), (a_idx(a_r0, a_c0+1), 1)],
            [(b_idx(b_r0+1, b_c0+1), 1)],
            [(c_idx(c_r0, c_c0), -1), (c_idx(c_r0, c_c0+1), 1)]
        )
        # M6
        add_term(
            [(a_idx(a_r0+1, a_c0), 1), (a_idx(a_r0, a_c0), -1)],
            [(b_idx(b_r0, b_c0), 1), (b_idx(b_r0, b_c0+1), 1)],
            [(c_idx(c_r0+1, c_c0+1), 1)]
        )
        # M7
        add_term(
            [(a_idx(a_r0, a_c0+1), 1), (a_idx(a_r0+1, a_c0+1), -1)],
            [(b_idx(b_r0+1, b_c0), 1), (b_idx(b_r0+1, b_c0+1), 1)],
            [(c_idx(c_r0, c_c0), 1)]
        )

    def add_outer_block(a_r0, a_c, b_r, c_r0, c_c0, num_cols):
        # A is 2x1 at (a_r0, a_c)
        # B is 1x4 at (b_r, c_c0)
        # C is 2x4 at (c_r0, c_c0)
        for i in range(2):
            for k in range(num_cols):
                add_term(
                    [(a_idx(a_r0 + i, a_c), 1)],
                    [(b_idx(b_r, c_c0 + k), 1)],
                    [(c_idx(c_r0 + i, c_c0 + k), 1)]
                )

    # Left Half (k=0..3)
    # A_L (2x2) at (0,0), B_L (2x4) at (0,0)
    # Split B_L into B_L1 (2x2) at (0,0) and B_L2 (2x2) at (0,2)
    
    # M222 Block 1: A(0,0) 2x2, B(0,0) 2x2, C(0,0) 2x2
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # M222 Block 2: A(0,0) 2x2, B(0,2) 2x2, C(0,2) 2x2
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Outer Block: A(0,2) 2x1, B(2,0) 1x4, C(0,0) 2x4
    add_outer_block(0, 2, 2, 0, 0, 4)

    # Right Half (k=4..7)
    # A_L (2x2) at (0,0), B_R (2x4) at (0,4)
    # Split B_R into B_R1 (2x2) at (0,4) and B_R2 (2x2) at (0,6)
    
    # M222 Block 3: A(0,0) 2x2, B(0,4) 2x2, C(0,4) 2x2
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # M222 Block 4: A(0,0) 2x2, B(0,6) 2x2, C(0,6) 2x2
    add_strassen_block(0, 0, 0, 6, 0, 6)
    
    # Outer Block: A(0,2) 2x1, B(2,4) 1x4, C(0,4) 2x4
    add_outer_block(0, 2, 2, 0, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
