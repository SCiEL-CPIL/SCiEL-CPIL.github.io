"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Use a recursive Strassen decomposition (rank 49) for the first four
columns of B (4x4*4x4), and a block Strassen approach for the last two columns
(rank 28). Total rank: 77.

The 4x4*4x4 part uses 7 outer Strassen products, each a 2x2*2x2 Strassen
(rank 7), giving 49 terms. The 4x4*4x2 part splits into 4 independent
2x2*2x2 Strassen products (one per block pair), giving 28 terms.
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

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            # U part: 2x2 block of A
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            # V part: 2x2 block of B
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            # W part: 2x2 block of C
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    # Quick self-verification on a random instance
    if time.time() < deadline - 5.0:
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        C_expected = A_test @ B_test
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Reconstruct C from factors: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k... 
        # Actually the convention is:
        # U has shape (n*m, rank) = (16, rank), indexed by i*m+j
        # V has shape (m*p, rank) = (24, rank), indexed by j*p+k
        # W has shape (p*n, rank) = (24, rank), indexed by k*n+i
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_recon = np.zeros((4, 6))
        for i in range(4):
            for k in range(6):
                s = 0.0
                for j in range(4):
                    for r in range(total_rank):
                        s += U_arr[i * 4 + j, r] * V_arr[j * 6 + k, r] * W_arr[k * 4 + i, r]
                C_recon[i, k] = s
        err = np.max(np.abs(C_recon - C_expected))
        if err > 1e-6:
            # Fallback: return a valid but higher-rank decomposition (schoolbook, rank 96)
            U_fb = np.zeros((16, 96))
            V_fb = np.zeros((24, 96))
            W_fb = np.zeros((24, 96))
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        r = i * 24 + j * 6 + k
                        U_fb[i * 4 + j, r] = 1
                        V_fb[j * 6 + k, r] = 1
                        W_fb[k * 4 + i, r] = 1
            return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
