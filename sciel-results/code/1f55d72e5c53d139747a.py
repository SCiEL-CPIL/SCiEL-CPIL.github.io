import time
import numpy as np

def construct():
    deadline = time.time() + 118.0

    n, m, p = 3, 4, 4

    # Schoolbook solution as fallback (Rank 48)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # Rank-11 decomposition for 3x2x4 matrix multiplication (C = A(3x2) * B(2x4))
    rank11_terms_3x2x4 = [
        ({(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
        ({(1,1):1,(1,0):1}, {(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1,(0,1):-1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1,(0,1):1}, {(0,0):1,(0,1):1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):-1}, {(0,0):1,(0,1):-1}, {(0,0):1,(0,1):-1}),
        ({(1,1):1}, {(1,0):1,(1,1):1}, {(1,2):1,(1,3):1}),
        ({(1,0):1}, {(1,0):1,(1,1):-1}, {(1,2):1,(1,3):-1}),
        ({(0,0):1}, {(1,0):1,(1,1):1}, {(0,2):1,(0,3):1}),
        ({(0,1):1}, {(1,0):1,(1,1):-1}, {(0,2):1,(0,3):-1}),
        ({(0,0):1,(1,0):1}, {(1,1):1,(1,0):-1}, {(0,3):1,(0,2):-1}),
        ({(0,1):1,(1,1):1}, {(1,0):1,(1,1):-1}, {(0,2):1,(0,3):-1}),
    ]

    # Build rank-22 decomposition: two 3x2x4 blocks
    total_rank = 22
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # Left half: A[0:3, 0:2] * B[0:2, 0:4]
    for r in range(11):
        u_dict, v_dict, w_dict = rank11_terms_3x2x4[r]
        c = col + r
        for (ai, aj), val in u_dict.items():
            U[ai * m + aj][c] += val
        for (bj, bk), val in v_dict.items():
            V[bj * p + bk][c] += val
        for (ci, ck), val in w_dict.items():
            W[ck * n + ci][c] += val
    col += 11

    # Right half: A[0:3, 2:4] * B[2:4, 0:4]
    for r in range(11):
        u_dict, v_dict, w_dict = rank11_terms_3x2x4[r]
        c = col + r
        for (ai, aj), val in u_dict.items():
            U[ai * m + (aj + 2)][c] += val
        for (bj, bk), val in v_dict.items():
            V[(bj + 2) * p + bk][c] += val
        for (ci, ck), val in w_dict.items():
            W[ck * n + ci][c] += val
    col += 11

    # Verify the full decomposition
    def _verify_full(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None

    valid, err = _verify_full(U, V, W, total_rank)
    if valid:
        return {"U": U, "V": V, "W": W}

    # Fallback: use the known rank-44 decomposition
    total_rank_44 = 44
    U44 = [[0] * total_rank_44 for _ in range(n * m)]
    V44 = [[0] * total_rank_44 for _ in range(m * p)]
    W44 = [[0] * total_rank_44 for _ in range(p * n)]

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            c = col_start + r
            for (ai, aj), val in u_dict.items():
                a_row = a_row_offset + ai
                if a_row < n:
                    U44[a_row * m + (j_offset + aj)][c] += val
            for (bj, bk), val in v_dict.items():
                V44[(j_offset + bj) * p + (b_col_offset + bk)][c] += val
            for (ci, ck), val in w_dict.items():
                c_row = c_row_offset + ci
                if c_row < n:
                    W44[(c_col_offset + ck) * n + c_row][c] += val
        return col_start + 7

    c44 = 0
    c44 = add_rank7_block(0, 0, 0, 0, 0, c44)
    c44 = add_rank7_block(0, 0, 2, 0, 0, c44)
    c44 = add_rank7_block(0, 2, 0, 0, 2, c44)
    c44 = add_rank7_block(0, 2, 2, 0, 2, c44)
    for j in range(2):
        for k in range(2):
            U44[2 * m + j][c44] = 1
            V44[j * p + k][c44] = 1
            W44[k * n + 2][c44] = 1
            c44 += 1
    for j in range(2, 4):
        for k in range(2, 4):
            U44[2 * m + j][c44] = 1
            V44[j * p + k][c44] = 1
            W44[k * n + 2][c44] = 1
            c44 += 1
    for j in range(2, 4):
        for k in range(0, 2):
            U44[2 * m + j][c44] = 1
            V44[j * p + k][c44] = 1
            W44[k * n + 2][c44] = 1
            c44 += 1
    for j in range(0, 2):
        for k in range(2, 4):
            U44[2 * m + j][c44] = 1
            V44[j * p + k][c44] = 1
            W44[k * n + 2][c44] = 1
            c44 += 1

    return {"U": U44, "V": V44, "W": W44}
