"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Use a recursive Strassen decomposition for the 4x4*4x4 part (rank 49)
and a block-based approach for the 4x4*4x2 part.

Key insight: The 4x4*4x4 multiplication can be decomposed using 7 outer Strassen
products, each of which is itself a 2x2*2x2 Strassen (rank 7), giving 7*7 = 49.

For the 4x4*4x2 part, we split into 4 block products of 2x2*2x2, each using
Strassen (rank 7), giving 4*7 = 28.

Total rank: 49 + 28 = 77.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Total rank: 49 (4x4*4x4) + 28 (4x4*4x2) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))  # 16 x 77
    V = np.zeros((m * p, total_rank))  # 24 x 77
    W = np.zeros((p * n, total_rank))  # 24 x 77

    # Part 1: 4x4*4x4 Strassen for columns 0-3 of B
    # Recursive Strassen: 7 outer products, each is a 2x2*2x2 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            # U part: combine outer Strassen t with inner Strassen s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            # V part
            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            # W part
            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5 of B
    # Split into 4 block products, each 2x2*2x2 using Strassen (rank 7)
    # C_top = A_top[:, 0:2] * B_top + A_top[:, 2:4] * B_bottom
    # C_bottom = A_bottom[:, 0:2] * B_top + A_bottom[:, 2:4] * B_bottom
    # where B_top = B[0:2, 4:6], B_bottom = B[2:4, 4:6]

    block_products = [
        # (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
        (0, 0, 0, 4, 0, 4),   # A_top[:, 0:2] * B_top -> C_top
        (0, 2, 2, 4, 0, 4),   # A_top[:, 2:4] * B_bottom -> C_top
        (2, 0, 0, 4, 2, 4),   # A_bottom[:, 0:2] * B_top -> C_bottom
        (2, 2, 2, 4, 2, 4),   # A_bottom[:, 2:4] * B_bottom -> C_bottom
    ]

    # Strassen 2x2*2x2: 7 terms
    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            # U part: A[ar+i, ac+j] for i,j in 0,1
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            # V part: B[br+j, bc+k] for j,k in 0,1
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            # W part: C[cr+i, cc+k] for i,k in 0,1
            # W index is k*4 + i (trace convention W[k*n+i] with n=4)
            for i in range(2):
                for k in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
