import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 patterns
    # U, V, W entries are (index, value)
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
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    # 8 blocks: (i_off, j_off, k_off)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
        (0, 2, 6)
    ]
    
    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        base_rank = b_idx * 7
        for r in range(7):
            cr = base_rank + r
            for idx, val in sU[r]:
                li, lj = divmod(idx, 2)
                gi = li + i_off
                gj = lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in sV[r]:
                lj, lk = divmod(idx, 2)
                gj = lj + j_off
                gk = lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in sW[r]:
                lk, li = divmod(idx, 2)
                gk = lk + k_off
                gi = li + i_off
                W[gk * 2 + gi][cr] += val

    return {"U": U, "V": V, "W": W}
