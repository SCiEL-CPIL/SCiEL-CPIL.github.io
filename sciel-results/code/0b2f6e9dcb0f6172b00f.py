import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 4
    R = 28

    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen 7-term decomposition for 2x2x2 matrix multiplication
    # Each term: (list of (row, col, coeff) for U), (list for V), (list for W)
    # Local 2x2 indices: row 0,1 and col 0,1

    S = []

    # Term 0: (a+d)(e+h) contributes to C[0,0] and C[1,1]
    S.append(([
        (0, 0, 1), (0, 1, 1),   # U: a + d
        (0, 0, 1), (1, 1, 1),   # V: e + h
        (0, 0, 1), (1, 1, 1)    # W: C00 + C11
    ]))

    # Term 1: (c+d)*e contributes to C[1,0] and -C[1,1]
    S.append(([
        (1, 0, 1), (1, 1, 1),   # U: c + d
        (0, 0, 1),               # V: e
        (1, 0, 1), (1, 1, -1)   # W: C10 - C11
    ]))

    # Term 2: a*(f-h) contributes to C[0,1] and C[1,1]
    S.append(([
        (0, 0, 1),               # U: a
        (0, 1, 1), (1, 1, -1),  # V: f - h
        (0, 1, 1), (1, 1, 1)    # W: C01 + C11
    ]))

    # Term 3: d*(g-e) contributes to C[0,0] and C[1,0]
    S.append(([
        (0, 1, 1),               # U: d
        (1, 0, 1), (0, 0, -1),  # V: g - e
        (0, 0, 1), (1, 0, 1)    # W: C00 + C10
    ]))

    # Term 4: (a+b)*h contributes to -C[0,0] and C[0,1]
    S.append(([
        (0, 0, 1), (1, 0, 1),   # U: a + b
        (1, 1, 1),               # V: h
        (0, 0, -1), (0, 1, 1)   # W: -C00 + C01
    ]))

    # Term 5: (c-a)*(e+f) contributes to C[1,1]
    S.append(([
        (1, 0, 1), (0, 0, -1),  # U: c - a
        (0, 0, 1), (0, 1, 1),  # V: e + f
        (1, 1, 1)               # W: C11
    ]))

    # Term 6: (b-d)*(g+h) contributes to C[0,0]
    S.append(([
        (1, 0, 1), (0, 1, -1),  # U: b - d
        (1, 0, 1), (1, 1, 1),  # V: g + h
        (0, 0, 1)               # W: C00
    ]))

    # Four 2x2 sub-blocks of the 2x4 * 4x4 product
    # Block (bi, bk) where bi in {0,1} is row-block of A, bk in {0,1} is col-block of B
    offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]

    col_idx = 0
    for (j_off, k_off) in offsets:
        if time.time() > deadline:
            break
        for term in S:
            if time.time() > deadline:
                break
            u_list, v_list, w_list = term[0][0], term[0][1], term[0][2]

            for (lr, lc, c) in u_list:
                gi = lr
                gj = lc + j_off
                U[gi * m + gj, col_idx] += c

            for (lr, lc, c) in v_list:
                gj = lr + j_off
                gk = lc + k_off
                V[gj * p + gk, col_idx] += c

            for (lr, lc, c) in w_list:
                gk = lr + k_off
                gi = lc
                W[gk * n + gi, col_idx] += c

            col_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
