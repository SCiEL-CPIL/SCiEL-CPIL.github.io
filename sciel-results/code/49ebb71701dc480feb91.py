import numpy as np
import time

def construct():
    start = time.time()
    DEADLINE = 145.0
    n, m, p = 3, 4, 4

    # Rank-44 decomposition via four 3x2x2 blocks (each rank 11)
    # 3x2x2 = 2x2x2 (Strassen, rank 7) + 1x2x2 (direct, rank 4)

    U_loc = np.zeros((6, 11), dtype=float)
    V_loc = np.zeros((4, 11), dtype=float)
    W_loc = np.zeros((6, 11), dtype=float)

    # Strassen 2x2x2 on rows 0,1 of local A/C (local indices 0-3 for A/C, 0-3 for B)
    # S1 = (a00+a11)(b00+b11)
    U_loc[0, 0] = 1.0; U_loc[3, 0] = 1.0
    V_loc[0, 0] = 1.0; V_loc[3, 0] = 1.0
    W_loc[0, 0] = 1.0; W_loc[3, 0] = 1.0

    # S2 = (a10+a11)(b00)
    U_loc[2, 1] = 1.0; U_loc[3, 1] = 1.0
    V_loc[0, 1] = 1.0
    W_loc[2, 1] = 1.0; W_loc[3, 1] = -1.0

    # S3 = a00(b10-b00)
    U_loc[0, 2] = 1.0
    V_loc[1, 2] = 1.0; V_loc[3, 2] = -1.0
    W_loc[1, 2] = 1.0; W_loc[3, 2] = 1.0

    # S4 = a11(b01-b11)
    U_loc[3, 3] = 1.0
    V_loc[2, 3] = 1.0; V_loc[0, 3] = -1.0
    W_loc[0, 3] = 1.0; W_loc[2, 3] = 1.0

    # S5 = (a00+a01)(b11)
    U_loc[0, 4] = 1.0; U_loc[1, 4] = 1.0
    V_loc[3, 4] = 1.0
    W_loc[0, 4] = -1.0; W_loc[1, 4] = 1.0

    # S6 = (a10-a00)(b00+b10)
    U_loc[2, 5] = 1.0; U_loc[0, 5] = -1.0
    V_loc[0, 5] = 1.0; V_loc[1, 5] = 1.0
    W_loc[3, 5] = 1.0

    # S7 = (a01-a11)(b01+b11)
    U_loc[1, 6] = 1.0; U_loc[3, 6] = -1.0
    V_loc[2, 6] = 1.0; V_loc[3, 6] = 1.0
    W_loc[0, 6] = 1.0

    # Direct terms for row 2 (local indices 4,5 for A/C; B indices 0-3)
    # D1: a20*b00 -> c20
    U_loc[4, 7] = 1.0
    V_loc[0, 7] = 1.0
    W_loc[4, 7] = 1.0

    # D2: a21*b10 -> c20
    U_loc[5, 8] = 1.0
    V_loc[2, 8] = 1.0
    W_loc[4, 8] = 1.0

    # D3: a20*b01 -> c21
    U_loc[4, 9] = 1.0
    V_loc[1, 9] = 1.0
    W_loc[5, 9] = 1.0

    # D4: a21*b11 -> c21
    U_loc[5, 10] = 1.0
    V_loc[3, 10] = 1.0
    W_loc[5, 10] = 1.0

    # Assemble global 3x4x4 via four 3x2x2 blocks
    # A is 3x4, B is 4x4, C is 4x3 (stored as W[k*n+i] = C[i,k] where i=row, k=col)
    # Block (ac, br, bc): A[:, ac:ac+2] * B[br:br+2, bc:bc+2] -> C[:, bc:bc+2]
    blocks = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2),
    ]

    U = np.zeros((n * m, 44), dtype=float)
    V = np.zeros((m * p, 44), dtype=float)
    W = np.zeros((p * n, 44), dtype=float)

    for k, (ac, br, bc) in enumerate(blocks):
        off = k * 11

        # A: local (3,2) row-major -> global (3,4)
        for lr in range(3):
            for lc in range(2):
                li = lr * 2 + lc
                gi = lr * m + (ac + lc)
                U[gi, off:off + 11] += U_loc[li, :]

        # B: local (2,2) row-major -> global (4,4)
        for lr in range(2):
            for lc in range(2):
                li = lr * 2 + lc
                gi = (br + lr) * p + (bc + lc)
                V[gi, off:off + 11] += V_loc[li, :]

        # C: local (3,2) row-major -> global C stored as W[col * n + row]
        for lr in range(3):
            for lc in range(2):
                li = lr * 2 + lc
                gi = (bc + lc) * n + lr
                W[gi, off:off + 11] += W_loc[li, :]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
