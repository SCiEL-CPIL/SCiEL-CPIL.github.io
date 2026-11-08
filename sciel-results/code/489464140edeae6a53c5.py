import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    rank = 88

    u_rows = n * m
    v_rows = m * p
    w_rows = p * n

    U = np.zeros((u_rows, rank), dtype=int)
    V = np.zeros((v_rows, rank), dtype=int)
    W = np.zeros((w_rows, rank), dtype=int)

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                break
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1

    if time.time() <= deadline:
        # Part 2: i=1,2 using Strassen 2x2x2 blocks
        # Strassen matrices for 2x2x2
        S_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ])
        S_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        S_W = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ])

        # Reshape to (7, 2, 2)
        SU = S_U.reshape(7, 2, 2)
        SV = S_V.reshape(7, 2, 2)
        SW = S_W.reshape(7, 2, 2)

        for j_block in range(m // 2):
            for k_block in range(p // 2):
                for t in range(7):
                    if time.time() > deadline:
                        break

                    for li in range(2):
                        for lj in range(2):
                            val = SU[t, li, lj]
                            if val:
                                gi = 1 + li
                                gj = j_block * 2 + lj
                                U[gi * m + gj, col] += val

                    for lj in range(2):
                        for lk in range(2):
                            val = SV[t, lj, lk]
                            if val:
                                gj = j_block * 2 + lj
                                gk = k_block * 2 + lk
                                V[gj * p + gk, col] += val

                    for li in range(2):
                        for lk in range(2):
                            val = SW[t, li, lk]
                            if val:
                                gi = 1 + li
                                gk = k_block * 2 + lk
                                W[gk * n + gi, col] += val

                    col += 1

    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
