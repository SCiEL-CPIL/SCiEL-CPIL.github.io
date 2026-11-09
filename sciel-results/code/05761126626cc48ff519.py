import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 110

    n, m, p = 2, 4, 8
    rank = 56

    # Rank-7 Strassen decomposition for 2x2 * 2x2
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)

    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Build full rank-56 decomposition
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # 8 blocks: a=0,1 (A left/right), b=0,1 (B top/bottom)
    # Each block uses 7 rank-1 terms
    for a in range(2):
        for b in range(2):
            r0 = (a * 2 + b) * 7
            for s in range(7):
                r = r0 + s
                # U: rows for A block (a selects left/right half of A)
                # A has shape (2,4); block (a) uses cols a*2..a*2+1
                for i in range(2):
                    for jp in range(2):
                        U[i * m + a * 2 + jp, r] = U2[s, i * 2 + jp]

                # V: rows for B block (b selects top/bottom half of B)
                # B has shape (4,8); block (b) uses rows b*2..b*2+1
                for jp in range(2):
                    for kp in range(2):
                        V[(a * 2 + jp) * p + b * 2 + kp, r] = V2[s, jp * 2 + kp]

                # W: rows for output C (2,8)
                # C[i, b*2+kp] for i in 0,1; kp in 0,1
                for i in range(2):
                    for kp in range(2):
                        W[(b * 2 + kp) * n + i, r] = W2[i * 2 + kp, s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
