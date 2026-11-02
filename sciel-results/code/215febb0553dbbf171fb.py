import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    num_blocks = p // m  # 5 blocks of 2 columns

    rank = 7 * num_blocks

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 7 products for 2x2 x 2x2
    # Local U: 0=a,1=b,2=c,3=d
    # Local V: 0=e,1=f,2=g,3=h
    # Local W: 0=C00,1=C01,2=C10,3=C11
    strassen = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    for b in range(num_blocks):
        base = b * 7
        for k in range(7):
            r = base + k
            ui, uc, vi, vc, wi, wc = strassen[k]
            for i, c in zip(ui, uc):
                U[i, r] = c
            for i, c in zip(vi, vc):
                # local 0->row0,col2b; 1->row0,col2b+1; 2->row1,col2b; 3->row1,col2b+1
                row = i // 2
                col = 2 * b + (i % 2)
                V[row * p + col, r] = c
            for i, c in zip(wi, wc):
                # local 0->C00,1->C01,2->C10,3->C11
                # W index: W[k*n + r] where k is output col, r is output row
                # C00: k=2b, r=0 -> 2b*2+0 = 4b
                # C01: k=2b+1, r=0 -> (2b+1)*2+0 = 4b+2
                # C10: k=2b, r=1 -> 2b*2+1 = 4b+1
                # C11: k=2b+1, r=1 -> (2b+1)*2+1 = 4b+3
                out_k = i // 2
                out_r = i % 2
                W[(2 * b + out_k) * n + out_r, r] = c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
