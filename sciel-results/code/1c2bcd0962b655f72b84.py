"""Program for 2x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)

    # Strassen 2x2x2 terms
    # Each term: (U_dict, V_dict, W_dict)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    terms = [
        # M1 = (x00+x11)(y00+y11) -> z00 += M1, z11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (x10+x11)y00 -> z10 += M2, z11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = x00(y01-y11) -> z01 += M3, z11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = x11(y10-y00) -> z00 += M4, z10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (x00+x01)y11 -> z00 -= M5, z01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (x10-x00)(y00+y01) -> z11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (x01-x11)(y10+y11) -> z00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Block structure:
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]

    for b in range(4):
        a_col_offset = (b % 2) * 2       # 0 for blocks 0,2; 2 for blocks 1,3
        b_row_offset = (b % 2) * 2       # 0 for blocks 0,2; 2 for blocks 1,3
        b_col_offset = (b // 2) * 2      # 0 for blocks 0,1; 2 for blocks 2,3
        c_col_offset = (b // 2) * 2      # 0 for blocks 0,1; 2 for blocks 2,3

        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]

            for li, coeff in u_coeffs.items():
                li_row, li_col = li // 2, li % 2
                g_i = li_row
                g_j = a_col_offset + li_col
                g_idx = g_i * m + g_j  # i * 4 + j
                U[g_idx][col] += coeff

            for li, coeff in v_coeffs.items():
                li_row, li_col = li // 2, li % 2
                g_j = b_row_offset + li_row
                g_r = b_col_offset + li_col
                g_idx = g_j * p + g_r  # j * 4 + r
                V[g_idx][col] += coeff

            for li, coeff in w_coeffs.items():
                li_row, li_col = li // 2, li % 2
                g_i = li_row
                g_r = c_col_offset + li_col
                g_idx = g_r * n + g_i  # r * 2 + i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
