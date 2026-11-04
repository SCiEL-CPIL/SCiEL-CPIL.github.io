import time
import numpy as np

def construct():
    """
    Construct a rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for 2x2x2 blocks.
    Splits 4x4x6 into 4x4x4 (rank 49) and 4x4x2 (rank 28).
    """
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen factors for 2x2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    rank_222 = 7
    U_s = [[0.0] * rank_222 for _ in range(4)]
    V_s = [[0.0] * rank_222 for _ in range(4)]
    W_s = [[0.0] * rank_222 for _ in range(4)]

    # Strassen products
    # (U_block, V_block, W_block)
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
            U_s[i * 2 + j][m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k][m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i][m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # 1. 4x4x4 Part (columns 0-48)
    # Kronecker structure: U[i,j, t*7+s] = U_s[2I+J, t] * U_s[2i0+j0, s]
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val

    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val

    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Part (columns 49-76)
    # 4 products of 2x2x2 Strassen
    # A_IJ * B_part -> C_IK
    # B_part is B[2*is_bot + j0, 4 + k0]
    products = [
        (0, 0, 0), # I=0, J=0, Top (is_bot=0)
        (0, 1, 1), # I=0, J=1, Bot (is_bot=1)
        (1, 0, 0), # I=1, J=0, Top
        (1, 1, 1), # I=1, J=1, Bot
    ]

    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7

        # A part
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val

        # B part
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val

        # C part
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    return {"U": U, "V": V, "W": W}
