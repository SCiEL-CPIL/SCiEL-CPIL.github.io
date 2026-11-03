import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=float)
    V_triv = np.zeros((m * p, rank_triv), dtype=float)
    W_triv = np.zeros((p * n, rank_triv), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3

    # Strassen 2x2 definitions (7 multiplications)
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W2[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if not np.allclose(T_recon, T, atol=1e-9):
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    if time.time() > deadline - 5:
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
