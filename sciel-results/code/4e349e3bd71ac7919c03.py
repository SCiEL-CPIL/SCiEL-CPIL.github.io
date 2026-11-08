import numpy as np

def construct():
    n, m, p = 3, 4, 4

    # Rank-44 decomposition for M_{3,4,4}
    # Built from 4 blocks of a rank-11 M_{3,2,2} decomposition
    rank = 44

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Local 3x2x2 decomposition (Rank 11)
    # A: 3x2, B: 2x2, C: 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen-like part for rows 0,1 of A (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Extra part for Row 2 of A (cols 7-10)
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1

    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1

    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1

    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # Place the 4 blocks
    # Block (bi): A[:, a0:a0+2] * B[b0:b0+2, c0:c0+2] -> C[:, c0:c0+2]
    # a0 = (bi // 2) * 2
    # b0 = (bi % 2) * 2
    # c0 = (bi % 2) * 2

    for bi in range(4):
        a0 = (bi // 2) * 2
        b0 = (bi % 2) * 2
        c0 = (bi % 2) * 2
        col_offset = bi * 11

        # A block: rows 0..2, cols a0..a0+1
        for r in range(3):
            for c in range(2):
                local_a = r * 2 + c
                global_a = r * m + (a0 + c)
                U[global_a, col_offset:col_offset + 11] += U_loc[local_a, :]

        # B block: rows b0..b0+1, cols c0..c0+1
        for r in range(2):
            for c in range(2):
                local_b = r * 2 + c
                global_b = (b0 + r) * p + (c0 + c)
                V[global_b, col_offset:col_offset + 11] += V_loc[local_b, :]

        # C block: rows 0..2, cols c0..c0+1
        for r in range(3):
            for c in range(2):
                local_c = r * 2 + c
                global_c = (c0 + c) * n + r
                W[global_c, col_offset:col_offset + 11] += W_loc[local_c, :]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
