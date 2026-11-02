import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 20s margin

    # Strassen 2x2x2 decomposition components
    # U_strassen[t] is 2x2 for A, V_strassen[t] is 2x2 for B, W_strassen[t] is 2x2 for C
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    U_strassen = np.array([
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: A21-A11
        [0, 1, 0, -1]   # M7: A12-A22
    ], dtype=float)

    V_strassen = np.array([
        [1, 0, 0, 1],   # M1: B11+B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B21-B22
        [-1, 0, 1, 0],  # M4: B12-B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11+B12
        [0, 0, 1, 1]    # M7: B21+B22
    ], dtype=float)

    W_strassen = np.array([
        [1, 0, 0, 1],   # M1: C11+C22
        [0, 0, 1, -1],  # M2: C21-C12
        [0, 1, 0, 1],   # M3: C12+C22
        [1, 0, 1, 0],   # M4: C11+C21
        [-1, 1, 0, 0],  # M5: C11-C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0]    # M7: C11
    ], dtype=float)

    rank = 42
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)

    # 6 Strassen blocks:
    # Blocks 0-2: A_L (cols 0,1) * B_T1,B_T2,B_T3 (rows 0,1; cols 0-1, 2-3, 4-5)
    # Blocks 3-5: A_R (cols 2,3) * B_B1,B_B2,B_B3 (rows 2,3; cols 0-1, 2-3, 4-5)
    # C output: 2x6, rows 0-1, cols 0-5

    col_base = 0
    for block_idx in range(6):
        if time.time() > deadline:
            break

        if block_idx < 3:
            a_r0, a_c0 = 0, 0
            b_r0, b_c0 = 0, block_idx * 2
            c_r0, c_c0 = 0, block_idx * 2
        else:
            a_r0, a_c0 = 0, 2
            b_r0, b_c0 = 2, (block_idx - 3) * 2
            c_r0, c_c0 = 0, (block_idx - 3) * 2

        for t in range(7):
            rank_col = col_base + t

            for lr in range(2):
                for lc in range(2):
                    u_val = U_strassen[t][lr * 2 + lc]
                    if u_val != 0:
                        # A index: row a_r0+lr, col a_c0+lc -> flat index (a_r0+lr)*m + (a_c0+lc)
                        U[(a_r0 + lr) * m + (a_c0 + lc), rank_col] += u_val

                    v_val = V_strassen[t][lr * 2 + lc]
                    if v_val != 0:
                        # B index: row b_r0+lr, col b_c0+lc -> flat index (b_r0+lr)*p + (b_c0+lc)
                        V[(b_r0 + lr) * p + (b_c0 + lc), rank_col] += v_val

                    w_val = W_strassen[t][lr * 2 + lc]
                    if w_val != 0:
                        # C index: row c_r0+lr, col c_c0+lc
                        # W convention: W[k*n+i, r] where k=col, i=row
                        # So flat index for C(row, col) is col*n + row
                        W[(c_c0 + lc) * n + (c_r0 + lr), rank_col] += w_val

        col_base += 7

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
