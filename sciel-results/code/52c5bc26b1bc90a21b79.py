"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term_flex(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            if coeff != 0:
                W[idx][col] += coeff
        col += 1

    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        u_00 = (a_row_start + 0) * m + (a_col_start + 0)
        u_01 = (a_row_start + 0) * m + (a_col_start + 1)
        u_10 = (a_row_start + 1) * m + (a_col_start + 0)
        u_11 = (a_row_start + 1) * m + (a_col_start + 1)
        
        v_00 = (b_row_start + 0) * p + (b_col_start + 0)
        v_01 = (b_row_start + 0) * p + (b_col_start + 1)
        v_10 = (b_row_start + 1) * p + (b_col_start + 0)
        v_11 = (b_row_start + 1) * p + (b_col_start + 1)
        
        w_00 = (c_col_start + 0) * n + (c_row_start + 0)
        w_01 = (c_col_start + 1) * n + (c_row_start + 0)
        w_10 = (c_col_start + 0) * n + (c_row_start + 1)
        w_11 = (c_col_start + 1) * n + (c_row_start + 1)
        
        # M1
        add_term_flex({u_00: 1, u_11: 1}, {v_00: 1, v_11: 1}, {w_00: 1, w_11: 1})
        # M2
        add_term_flex({u_10: 1, u_11: 1}, {v_00: 1}, {w_10: 1})
        # M3
        add_term_flex({u_00: 1}, {v_01: 1, v_11: -1}, {w_01: 1})
        # M4
        add_term_flex({u_11: 1}, {v_10: 1, v_00: -1}, {w_10: 1})
        # M5
        add_term_flex({u_00: 1, u_01: 1}, {v_11: 1}, {w_00: -1, w_01: 1})
        # M6
        add_term_flex({u_10: 1, u_00: -1}, {v_00: 1, v_01: 1}, {w_11: 1})
        # M7
        add_term_flex({u_01: 1, u_11: -1}, {v_10: 1, v_11: 1}, {w_00: 1})

    # Left Part (Cols 0, 1)
    # 1. A[:, 0] B[0, 0:2]
    add_term_flex({0: 1, 3: 1, 6: 1}, {0: 1, 1: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    # 2. A[0, 1:3] B[1:3, 0:2]
    add_term_flex({1: 1}, {4: 1, 5: 1}, {0: 1, 3: 1})
    add_term_flex({2: 1}, {8: 1, 9: 1}, {0: 1, 3: 1})
    # 3. A[1:3, 1:3] B[1:3, 0:2]
    add_strassen(1, 1, 1, 0, 1, 0)
    
    # Right Part (Cols 2, 3)
    # 1. A[:, 0] B[0, 2:4]
    add_term_flex({0: 1, 3: 1, 6: 1}, {2: 1, 3: 1}, {6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1})
    # 2. A[0, 1:3] B[1:3, 2:4]
    add_term_flex({1: 1}, {6: 1, 7: 1}, {6: 1, 9: 1})
    add_term_flex({2: 1}, {10: 1, 11: 1}, {6: 1, 9: 1})
    # 3. A[1:3, 1:3] B[1:3, 2:4]
    add_strassen(1, 1, 1, 2, 1, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
