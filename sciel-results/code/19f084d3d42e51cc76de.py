"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Strassen 2x2: rank 7
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11) (B11 + B12)
    # P7 = (A12 - A22) (B21 + B22)

    strassen_products = [
        # P1 = (A11 + A22)(B11 + B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A21 + A22) B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3 = A11 (B12 - B22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4 = A22 (B21 - B11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5 = (A11 + A12) B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6 = (A21 - A11) (B11 + B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A12 - A22) (B21 + B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # 4 sub-blocks: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4: split into left (cols 0-1) and right (cols 2-3)
    # B is 4x8: split into top (rows 0-1) and bottom (rows 2-3)
    # C is 2x8: split into left (cols 0-1) and right (cols 2-3)
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2 * 2x4 is split into two 2x2 * 2x2 subproblems
    blocks = [
        (0, 0, 0, 0),   # A_left * B_top_left
        (0, 0, 2, 2),   # A_left * B_top_right
        (2, 2, 0, 0),   # A_right * B_bottom_left
        (2, 2, 2, 2),   # A_right * B_bottom_right
    ]

    for a_off, b_row_off, b_col_off, c_col_off in blocks:
        for u_c, v_c, w_c in strassen_products:
            for (li, lj), coeff in u_c.items():
                gi = li
                gj = a_off + lj
                U[gi * m + gj][col] += coeff
            for (lj, lk), coeff in v_c.items():
                gj = b_row_off + lj
                gk = b_col_off + lk
                V[gj * p + gk][col] += coeff
            for (lk, li), coeff in w_c.items():
                gk = c_col_off + lk
                gi = li
                W[gk * n + gi][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
