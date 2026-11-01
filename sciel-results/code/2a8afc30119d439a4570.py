import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    rank = 42

    # Pre-define Strassen factors
    # Local indices for 2x2: 0:00, 1:01, 2:10, 3:11
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for r in range(rank):
        # r = j_block * 21 + k_block * 7 + prod_idx
        # 2 j_blocks, 3 k_blocks, 7 prods
        rem = r
        prod_idx = rem % 7
        rem = rem // 7
        k_block = rem % 3
        j_block = rem // 3

        u_factors = sU[prod_idx]
        v_factors = sV[prod_idx]
        w_factors = sW[prod_idx]

        for li, val in u_factors:
            gi = li // 2
            gj = li % 2
            U[gi * m + (j_block * 2 + gj), r] = val

        for li, val in v_factors:
            gj = li // 2
            gk = li % 2
            V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = val

        for li, val in w_factors:
            gi = li // 2
            gk = li % 2
            W[(k_block * 2 + gk) * n + gi, r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
