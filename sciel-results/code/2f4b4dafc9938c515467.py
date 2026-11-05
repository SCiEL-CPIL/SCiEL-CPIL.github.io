import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    n, m, p = 4, 4, 6

    # Base Strassen factors for 2x2x2
    # U, V, W are 4x7
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    prods = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Part 1: 4x4 * 4x4 (Columns 0-3 of B)
    # This is 7 Strassen blocks, each of rank 7.
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            # U: A is 4x4. Block (I, J) is 2x2.
            # U_s[2*I+J, t] is the coefficient for block (I,J) in the t-th Strassen product for the outer block.
            # U_s[2*i0+j0, s] is the coefficient for element (i0,j0) in the inner block.
            # Wait, the current program's logic for Part 1 is:
            # val = U_s[2*I + J, t] * U_s[2*i0 + j0, s]
            # This effectively computes the Kronecker product of the Strassen factors.
            # This is correct for 4x4x4 using 2x2 blocks.
            
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4 * 4x2 (Columns 4-5 of B)
    # This is 4 independent 2x2x2 Strassen multiplications.
    # Each uses rank 7.
    # Block definitions:
    # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
    # C_col_start is the column offset in C (which is 4,5).
    # B_col_start is the column offset in B (which is 4,5).
    
    blocks_442 = [
        # (ar, ac, br, bc, cr, cc)
        (0, 0, 0, 4, 0, 4), # A11 * B_R1 -> C_R1
        (0, 2, 2, 4, 0, 4), # A12 * B_R2 -> C_R1
        (2, 0, 0, 4, 2, 4), # A21 * B_R1 -> C_R2
        (2, 2, 2, 4, 2, 4), # A22 * B_R2 -> C_R2
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks_442):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            # Fill V
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            # Fill W
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
