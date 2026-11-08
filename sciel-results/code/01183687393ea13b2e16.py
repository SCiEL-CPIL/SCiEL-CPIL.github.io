import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = 3, 4, 4

    # Local 3x2x2 decomposition template (rank 11)
    # A: 3x2, B: 2x2, C: 3x2
    # Strassen 2x2x2 (7 terms) + 4 direct terms for row 2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # S1 = (a00+a11)(b00+b11) -> c00, c11
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # S2 = (a10+a11)b00 -> c10, -c11
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # S3 = a00(b10-b00) -> c01, c11
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # S4 = a11(b01-b11) -> c00, c10
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # S5 = (a00+a01)b11 -> -c00, c01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # S6 = (a10-a00)(b00+b10) -> c11
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # S7 = (a01-a11)(b01+b11) -> c00
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # D1: a20*b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1

    # D2: a21*b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1

    # D3: a20*b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1

    # D4: a21*b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # Build full 3x4x4 decomposition: 4 blocks of 3x2x2, rank 44
    U = np.zeros((12, 44))
    V = np.zeros((16, 44))
    W = np.zeros((12, 44))

    # Block definitions: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # C row offset is always 0 since we use all 3 rows
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2)
    ]

    for bi, (ac, br, bc, cc) in enumerate(blocks):
        off = bi * 11

        # Map A (3x2 local) to global A (3x4)
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = r * 4 + (ac + c)
                U[gi, off:off+11] += U_loc[li, :]

        # Map B (2x2 local) to global B (4x4)
        for r in range(2):
            for c in range(2):
                li = r * 2 + c
                gi = (br + r) * 4 + (bc + c)
                V[gi, off:off+11] += V_loc[li, :]

        # Map C (3x2 local) to global C (3x4), W index = col*3 + row
        for r in range(3):
            for c in range(2):
                li = r * 2 + c
                gi = (cc + c) * 3 + r
                W[gi, off:off+11] += W_loc[li, :]

    if time.time() > DEADLINE - 5:
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
