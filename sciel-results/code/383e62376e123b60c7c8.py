import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    rank = 88

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U = np.zeros((u_rows, rank), dtype=float)
    V = np.zeros((v_rows, rank), dtype=float)
    W = np.zeros((w_rows, rank), dtype=float)

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            U[j, col] = 1
            V[j * p + k, col] = 1
            W[k * n, col] = 1
            col += 1
        if time.time() > deadline:
            break

    if time.time() <= deadline:
        # Part 2: i=1,2 using Strassen 2x2x2 blocks
        strassen_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ], dtype=float)
        strassen_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=float)
        strassen_W = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ], dtype=float)

        for j_block in range(m // 2):
            for k_block in range(p // 2):
                for t in range(7):
                    if time.time() > deadline:
                        break

                    # U part: rows i=1,2
                    for li in range(2):
                        for lj in range(2):
                            loc = li * 2 + lj
                            val = strassen_U[t, loc]
                            if val != 0:
                                gi = 1 + li
                                gj = j_block * 2 + lj
                                U[gi * m + gj, col] += val

                    # V part
                    for lj in range(2):
                        for lk in range(2):
                            loc = lj * 2 + lk
                            val = strassen_V[t, loc]
                            if val != 0:
                                gj = j_block * 2 + lj
                                gk = k_block * 2 + lk
                                V[gj * p + gk, col] += val

                    # W part
                    for li in range(2):
                        for lk in range(2):
                            loc = li * 2 + lk
                            val = strassen_W[t, loc]
                            if val != 0:
                                gi = 1 + li
                                gk = k_block * 2 + lk
                                W[gk * n + gi, col] += val

                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
