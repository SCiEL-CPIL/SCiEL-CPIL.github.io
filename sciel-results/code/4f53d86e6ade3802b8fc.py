"""Program for 5x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # Partition 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)

    U_terms = []
    V_terms = []
    W_terms = []

    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]

                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]

                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    # M1 = (A00+A11)(B00+B11)
                    U_terms.append([(a_row_start + 0, a_col_start + 0, 1), (a_row_start + 1, a_col_start + 1, 1)])
                    V_terms.append([(b_row_start + 0, b_col_start + 0, 1), (b_row_start + 0, b_col_start + 1, 1)])
                    W_terms.append([(c_row_start + 0, c_col_start + 0, 1), (c_row_start + 1, c_col_start + 1, 1)])

                    # M2 = (A10+A11)(B00)
                    U_terms.append([(a_row_start + 1, a_col_start + 0, 1), (a_row_start + 1, a_col_start + 1, 1)])
                    V_terms.append([(b_row_start + 0, b_col_start + 0, 1)])
                    W_terms.append([(c_row_start + 1, c_col_start + 0, 1), (c_row_start + 1, c_col_start + 1, -1)])

                    # M3 = (A00)(B01-B11)
                    U_terms.append([(a_row_start + 0, a_col_start + 0, 1)])
                    V_terms.append([(b_row_start + 0, b_col_start + 1, 1), (b_row_start + 1, b_col_start + 1, -1)])
                    W_terms.append([(c_row_start + 0, c_col_start + 1, 1), (c_row_start + 1, c_col_start + 1, 1)])

                    # M4 = (A11)(B10-B00)
                    U_terms.append([(a_row_start + 1, a_col_start + 1, 1)])
                    V_terms.append([(b_row_start + 1, b_col_start + 0, 1), (b_row_start + 0, b_col_start + 0, -1)])
                    W_terms.append([(c_row_start + 0, c_col_start + 0, 1), (c_row_start + 1, c_col_start + 0, 1)])

                    # M5 = (A00+A01)(B11)
                    U_terms.append([(a_row_start + 0, a_col_start + 0, 1), (a_row_start + 0, a_col_start + 1, 1)])
                    V_terms.append([(b_row_start + 1, b_col_start + 1, 1)])
                    W_terms.append([(c_row_start + 0, c_col_start + 0, -1), (c_row_start + 0, c_col_start + 1, 1)])

                    # M6 = (A10-A00)(B00+B01)
                    U_terms.append([(a_row_start + 1, a_col_start + 0, 1), (a_row_start + 0, a_col_start + 0, -1)])
                    V_terms.append([(b_row_start + 0, b_col_start + 0, 1), (b_row_start + 0, b_col_start + 1, 1)])
                    W_terms.append([(c_row_start + 1, c_col_start + 1, 1)])

                    # M7 = (A01-A11)(B10+B11)
                    U_terms.append([(a_row_start + 0, a_col_start + 1, 1), (a_row_start + 1, a_col_start + 1, -1)])
                    V_terms.append([(b_row_start + 1, b_col_start + 0, 1), (b_row_start + 1, b_col_start + 1, 1)])
                    W_terms.append([(c_row_start + 0, c_col_start + 0, 1)])

                else:
                    # Schoolbook for mixed-size blocks
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([(a_row_start + ii, a_col_start + jj, 1)])
                                V_terms.append([(b_row_start + jj, b_col_start + kk, 1)])
                                W_terms.append([(c_row_start + ii, c_col_start + kk, 1)])

    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for entry in U_terms[col]:
            row_idx = entry[0]
            val = entry[2]
            U[row_idx][col] = val
        for entry in V_terms[col]:
            row_idx = entry[0]
            val = entry[2]
            V[row_idx][col] = val
        for entry in W_terms[col]:
            row_idx = entry[0]
            val = entry[2]
            W[row_idx][col] = val

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    max_check = min(n, 5)
    for i in range(max_check):
        for j in range(max_check):
            for k in range(max_check):
                lhs = 0.0
                for r in range(rank):
                    lhs += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(lhs - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][column] = 1
                                V_fb[jj * p + kk][column] = 1
                                W_fb[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
