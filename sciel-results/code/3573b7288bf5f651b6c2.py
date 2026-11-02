import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank 56 decomposition using 8 Strassen blocks
    total_rank = 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    # Strassen local factors (verified from working code)
    # r, U_indices, U_vals, V_indices, V_vals, W_indices, W_vals
    strassen_factors = [
        # r=0
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # r=1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # r=2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # r=3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # r=4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # r=5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # r=6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    group = 0
    for half in range(2):  # 0: A[:, 0:2] & B[0:2, :], 1: A[:, 2:4] & B[2:4, :]
        for jj in range(p // 2):  # 4 column blocks of size 2
            base_r = group * 7
            group += 1

            # A half 'half': columns 2*half, 2*half+1
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]

            # B half 'half': rows 2*half, 2*half+1. Cols 2*jj, 2*jj+1.
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]

            # C output for this block: rows 0,1. Cols 2*jj, 2*jj+1.
            # W index = k*n + i
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_factors[r]
                
                for idx, val in zip(u_idx, u_val):
                    U[a_map[idx]][rr] += val
                for idx, val in zip(v_idx, v_val):
                    V[b_map[idx]][rr] += val
                for idx, val in zip(w_idx, w_val):
                    W[c_map[idx]][rr] += val

    # Deadline guard
    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
