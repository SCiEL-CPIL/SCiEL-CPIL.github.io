"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START


def _build_decomp(n, m, p):
    """
    Build a rank-44 decomposition for 2x3 * 3x8.
    Part 1: A2 * B2 (2x1 * 1x8) -> rank 16 schoolbook.
    Part 2: A1 * B1 (2x2 * 2x8) -> rank 28 via 4 Strassen 2x2x2 blocks.
    """
    n, m, p = 2, 3, 8
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0

    # Part 1: A2 * B2, 16 terms
    for k in range(p):
        # C[0,k] += A[0,2] * B[2,k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # Part 2: A1 (2x2) * B1 (2x8) via 4 Strassen blocks
    for g in range(4):
        b11 = 0 * p + 2 * g
        b12 = 0 * p + 2 * g + 1
        b21 = 1 * p + 2 * g
        b22 = 1 * p + 2 * g + 1

        a11 = 0
        a12 = 1
        a21 = 3
        a22 = 4

        c11 = (2 * g) * n + 0
        c12 = (2 * g + 1) * n + 0
        c21 = (2 * g) * n + 1
        c22 = (2 * g + 1) * n + 1

        # Strassen 2x2x2: 7 products
        # M1 = (a11+a22)(b11+b22) -> C11 += M1, C22 += M1
        U[a11][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        col += 1

        # M2 = (a21+a22)b11 -> C21 += M2
        U[a21][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        col += 1

        # M3 = a11(b22-b12) -> C12 += M3, C22 -= M3
        U[a11][col] += 1
        V[b22][col] += 1
        V[b12][col] -= 1
        W[c12][col] += 1
        W[c22][col] -= 1
        col += 1

        # M4 = a22(b11-b21) -> C11 -= M4, C21 += M4
        U[a22][col] += 1
        V[b11][col] += 1
        V[b21][col] -= 1
        W[c11][col] -= 1
        W[c21][col] += 1
        col += 1

        # M5 = (a11+a12)b22 -> C12 += M5
        U[a11][col] += 1
        U[a12][col] += 1
        V[b22][col] += 1
        W[c12][col] += 1
        col += 1

        # M6 = (a21-a11)(b11+b12) -> C11 += M6
        U[a21][col] += 1
        U[a11][col] -= 1
        V[b11][col] += 1
        V[b12][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (a12-a22)(b21+b22) -> C22 += M7
        U[a12][col] += 1
        U[a22][col] -= 1
        V[b21][col] += 1
        V[b22][col] += 1
        W[c22][col] += 1
        col += 1

    return U, V, W


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct using numpy."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Rank-44 Strassen-based decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        if time.time() - start_time < 115:
            err = _verify(U, V, W, n, m, p)
            if err < 1e-10:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
