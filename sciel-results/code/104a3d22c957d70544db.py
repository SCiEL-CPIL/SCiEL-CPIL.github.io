"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # Rank 50: 2 j-blocks * (3 Strassen 2x2x2 blocks of rank 7 + 1 schoolbook 2x2x1 block of rank 4)
    # = 2 * (21 + 4) = 50
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add Strassen 2x2x2 decomposition for C[k0:k0+2, i0:i0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        nonlocal col
        if time.time() > deadline:
            return False
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> contributes to c00, c11
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e -> contributes to c10, c11
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h) -> contributes to c01, c11
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e) -> contributes to c00, c10
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h -> contributes to c00, c01
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> contributes to c11
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> contributes to c00
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        return True

    def add_schoolbook_221_block(i0, j0, k0):
        """Add schoolbook 2x2x1 decomposition for C[k0, i0:i0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]"""
        nonlocal col
        if time.time() > deadline:
            return False
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e -> c00
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g -> c00
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e -> c10
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g -> c10
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        return True

    # j pairs: {0,1} and {2,3}
    # k pairs: {0,1}, {2,3}, {4,5}, and single {6}
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            if not add_strassen_block(0, j_start, k_start):
                break
        else:
            if not add_schoolbook_221_block(0, j_start, 6):
                break

    if col != rank:
        raise RuntimeError(f"Construction error: expected rank {rank}, got {col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
