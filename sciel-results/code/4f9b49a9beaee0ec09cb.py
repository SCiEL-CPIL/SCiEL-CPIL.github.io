import numpy as np

def construct():
    n, m, p = 2, 4, 8

    # Strassen-like 7-term decomposition for 2x2 * 2x2
    # A = [[a1, a2], [a3, a4]], B = [[b1, b2], [b3, b4]]
    # C = [[c1, c2], [c3, c4]]
    #
    # m1 = (a1+a4)(b1+b4)
    # m2 = (a3+a4)b1
    # m3 = a1(b2-b4)
    # m4 = a4(b3-b1)
    # m5 = (a1+a2)b4
    # m6 = (a3-a1)(b1+b2)
    # m7 = (a2-a4)(b3+b4)
    #
    # c1 = m1 - m4 + m7
    # c2 = m3 + m5
    # c3 = m2 + m4
    # c4 = m1 - m2 + m3 + m6

    # U coefficients for A (2x2 -> 4 entries)
    x_forms = np.array([
        [1, 0, 0, 1],    # m1: a1+a4
        [0, 0, 1, 1],    # m2: a3+a4
        [1, 0, 0, 0],    # m3: a1
        [0, 0, 0, 1],    # m4: a4
        [1, 1, 0, 0],    # m5: a1+a2
        [-1, 0, 1, 0],   # m6: a3-a1
        [0, 1, 0, -1]    # m7: a2-a4
    ])

    # V coefficients for B (2x2 -> 4 entries)
    y_forms = np.array([
        [1, 0, 0, 1],    # m1: b1+b4
        [1, 0, 0, 0],    # m2: b1
        [0, 1, 0, -1],   # m3: b2-b4
        [-1, 0, 1, 0],   # m4: b3-b1
        [0, 0, 0, 1],    # m5: b4
        [1, 1, 0, 0],    # m6: b1+b2
        [0, 0, 1, 1]     # m7: b3+b4
    ])

    # W coefficients for C (2x2 -> 4 entries)
    z_forms = np.array([
        [1, 0, 0, 1, -1, 0, 1],    # c1: m1 - m4 + m7
        [0, 0, 1, 0, 1, 0, 0],     # c2: m3 + m5
        [0, 1, 0, 1, 0, 0, 0],     # c3: m2 + m4
        [1, -1, 1, 0, 0, 1, 0]     # c4: m1 - m2 + m3 + m6
    ])

    # Decompose 2x4 * 4x8 into blocks
    # A is 2x4, split into 2 blocks of 2x2 (columns 0-1, 2-3)
    # B is 4x8, split into 4 blocks of 2x2 (rows 0-1, cols 0-1; rows 0-1, cols 2-3; etc.)
    # Actually, B is 4x8. We can think of it as 2 row-blocks of 2x8, or 4 col-blocks of 4x2.
    # Let's use: A has 2 col-blocks of 2x2. B has 4 col-blocks of 4x2, but we need 2x2 blocks.
    # B[j, k] where j in 0..3, k in 0..7.
    # For Strassen, we need 2x2 blocks.
    # A[i, j] where i in 0..1, j in 0..3.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # Let's partition:
    # A = [A1, A2] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3)
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # Wait, B is 4x8. So B has 4 rows and 8 cols.
    # Let's partition B into 2 row-blocks of 2x8, and each 2x8 into 4 col-blocks of 2x2.
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    # B_top = [B11, B12, B13, B14] (each 2x2)
    # B_bottom = [B21, B22, B23, B24] (each 2x2)
    #
    # C = A1 * B_top + A2 * B_bottom
    #   = A1 * [B11, B12, B13, B14] + A2 * [B21, B22, B23, B24]
    #   = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23, A1*B14 + A2*B24]
    #
    # Each term A1*Bjk and A2*Bjk is a 2x2 * 2x2 product, which we can do with 7 terms.
    # So total rank = 4 (col blocks) * 2 (row blocks) * 7 (Strassen) = 56

    num_bcol_blocks = 4  # B has 4 col-blocks of 2x2
    num_brow_blocks = 2  # B has 2 row-blocks of 2x2
    num_strassen = 7

    rank = num_bcol_blocks * num_brow_blocks * num_strassen

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    r = 0
    for g in range(num_bcol_blocks):  # column block of B
        for b in range(num_brow_blocks):  # row block of B
            for s in range(num_strassen):  # Strassen term
                # U: A is 2x4. A[i, j]
                # A1 = A[:, 0:2], A2 = A[:, 2:4]
                # For b=0 (A1), use cols 0,1. For b=1 (A2), use cols 2,3.
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = x_forms[s, i * 2 + jp]

                # V: B is 4x8. B[j, k]
                # B has 2 row-blocks (b) and 4 col-blocks (g).
                # Row block b: rows 2*b to 2*b+1
                # Col block g: cols 2*g to 2*g+1
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = y_forms[s, jp * 2 + kp]

                # W: C is 8x2 (transposed). C[k, i] where k in 0..7, i in 0..1
                # C[k, i] = sum_j A[i, j] * B[j, k]
                # For col block g, k = 2*g to 2*g+1
                # C[2*g+kp, i] += contribution from this Strassen term
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = z_forms[i * 2 + kp, s]

                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
