import time

DIMENSIONS = [2, 2, 10]

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen rank-7 decomposition for 2x2x2
    # U vectors index A as [A00, A01, A10, A11]
    # V vectors index B as [B00, B01, B10, B11]
    # W vectors index C as [C00, C10, C01, C11] (j*n+i convention)
    S = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 1, 0, -1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, 1)),
        ((0, 0, 0, 1), (-1, 0, 1, 0), (1, 1, 0, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 0, 1, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]

    t = 0
    for blk in range(p // 2):
        for u4, v4, w4 in S:
            # Fill U: row index = i*m + k
            U[0][t] = u4[0]
            U[1][t] = u4[1]
            U[2][t] = u4[2]
            U[3][t] = u4[3]

            # Fill V: row index = k*p + j, block cols are 2*blk and 2*blk+1
            V[0 * p + 2 * blk][t] = v4[0]
            V[0 * p + 2 * blk + 1][t] = v4[1]
            V[1 * p + 2 * blk][t] = v4[2]
            V[1 * p + 2 * blk + 1][t] = v4[3]

            # Fill W: row index = j*n + i, block rows are 2*blk and 2*blk+1
            W[2 * blk * n + 0][t] = w4[0]
            W[2 * blk * n + 1][t] = w4[1]
            W[(2 * blk + 1) * n + 0][t] = w4[2]
            W[(2 * blk + 1) * n + 1][t] = w4[3]

            t += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
