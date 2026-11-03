"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen 2x2x2x2 decomposition coefficients
    # Each term r has U_coeffs (for 2x2 A), V_coeffs (for 2x2 B), W_coeffs (for 2x2 C)
    # U_coeffs[i,j], V_coeffs[j,k], W_coeffs[k,i]
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        # M2 = (X10+X11)*Y00
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1}},
        # M3 = X00*(Y01-Y11)
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1}},
        # M4 = X11*(Y10-Y00)
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        # M5 = (X00+X01)*Y11
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        # M6 = (X10-X00)*(Y00+Y01)
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        # M7 = (X01-X11)*(Y10+Y11)
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
    ]

    # 4 sub-problems: (a_row_offset, b_row_offset, c_col_offset)
    # a_row_offset: which 2 rows of A's columns (0 or 2)
    # b_row_offset: which 2 rows of B (0 or 2)
    # c_col_offset: which 2 cols of C (0 or 2)
    sub_problems = [
        (0, 0, 0),  # A_left * B_top_left -> C cols 0-1
        (0, 0, 2),  # A_left * B_top_right -> C cols 2-3
        (2, 2, 0),  # A_right * B_bottom_left -> C cols 0-1
        (2, 2, 2),  # A_right * B_bottom_right -> C cols 2-3
    ]

    col = 0
    for a_off, b_off, c_off in sub_problems:
        for term in strassen_terms:
            for (si, sj), val in term['U'].items():
                # A index: i = si, j = a_off + sj
                U[si * m + (a_off + sj)][col] = val
            for (sj, sk), val in term['V'].items():
                # B index: j = b_off + sj, k = sk
                V[(b_off + sj) * p + sk][col] = val
            for (sk, si), val in term['W'].items():
                # C index: k = c_off + sk, i = si
                W[(c_off + sk) * n + si][col] = val
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
