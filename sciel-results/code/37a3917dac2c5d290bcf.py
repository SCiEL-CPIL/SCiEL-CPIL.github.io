import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    # Build 2x2x2 Strassen factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    for s in range(7):
        for a, va in alpha[s].items():
            if va != 0:
                U_s[a, s] += va
        for b, vb in beta[s].items():
            if vb != 0:
                V_s[b, s] += vb
        for (i, k, c) in recon[s]:
            W_s[k * 2 + i, s] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4 * 4x4 via Kronecker of Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for I in range(2):
                for J in range(2):
                    vt = U_s[2 * I + J, t]
                    if vt == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            vs = U_s[2 * i0 + j0, s]
                            if vs == 0:
                                continue
                            val = vt * vs
                            i = 2 * I + i0
                            j = 2 * J + j0
                            U[i * 4 + j, col] += val

            for J in range(2):
                for K in range(2):
                    vt = V_s[2 * J + K, t]
                    if vt == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            vs = V_s[2 * j0 + k0, s]
                            if vs == 0:
                                continue
                            val = vt * vs
                            j = 2 * J + j0
                            k = 2 * K + k0
                            V[j * 6 + k, col] += val

            for K in range(2):
                for I in range(2):
                    vt = W_s[2 * K + I, t]
                    if vt == 0:
                        continue
                    for k0 in range(2):
                        for i0 in range(2):
                            vs = W_s[2 * k0 + i0, s]
                            if vs == 0:
                                continue
                            val = vt * vs
                            k = 2 * K + k0
                            i = 2 * I + i0
                            W[k * 4 + i, col] += val

    # Part 2: 4x4 * 4x2 via 4 blocks of 2x2 * 2x2
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4)
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

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                xr = round(x)
                if abs(x - xr) < 1e-9:
                    new_row.append(int(xr))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
