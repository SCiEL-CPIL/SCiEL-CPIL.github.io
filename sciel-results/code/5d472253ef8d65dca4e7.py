import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    # --- Strassen 2x2*2x2 factors (7 factors) ---
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

    # Build 4x4*4x4 Strassen (rank 49) and 4x4*4x2 block Strassen (rank 28)
    # Total rank = 77

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((16, total_rank))  # n*m = 4*4
    V = np.zeros((24, total_rank))  # m*p = 4*6
    W = np.zeros((24, total_rank))  # p*n = 6*4

    # Part 1: 4x4 * 4x4 using two-level Strassen (rank 49)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
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

    # Part 2: 4x4 * 4x2 (the extra two columns of the 4x6 matrix)
    # Split into four 2x2 blocks: rows 0-1 & 2-3 of A, cols 0-1 & 2-3 of B_extra
    # Each 2x2*2x2 uses Strassen rank 7, so 4 blocks * 7 = 28
    block_products = [
        (0, 0, 0, 4, 0, 4),  # (ar, ac, br, bc, cr, cc)
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    # Check deadline
    if time.time() > deadline - 5.0:
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Convert to lists with integer rounding where appropriate
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    r.append(int(round(x)))
                else:
                    r.append(float(x))
            result.append(r)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
