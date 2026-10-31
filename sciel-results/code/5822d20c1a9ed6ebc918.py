"""Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: build the 4x4 multiplication from a 2x2-block Strassen-style algorithm
expanded into scalar rank-1 terms, combine like terms, verify exactly, and fall
back to schoolbook (rank 64) if verification fails. Target rank 49 (de Groote).
"""
import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    n = m = p = 4
    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    rank = Um.shape[1]
    U3 = Um.reshape(n, m, rank)
    V3 = Vm.reshape(m, p, rank)
    W3 = Wm.reshape(p, n, rank)
    rng = np.random.default_rng(12345)
    for _ in range(4):
        A = rng.integers(-4, 5, size=(n, m))
        B = rng.integers(-4, 5, size=(m, p))
        C = A @ B
        Cc = np.einsum('ijr,jkr,ki->ik', U3, V3, W3)
        if not np.array_equal(Cc, C):
            return False
    return True


def _schoolbook():
    n = m = p = 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _block_strassen():
    """Build 4x4 mult from 2x2-block Strassen, expanded to scalar rank-1 terms."""
    n = m = p = 4
    terms = defaultdict(int)

    def macro(Xr, Xc, Yr, Yc, Cr, Cc, s):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    i = 2 * Cr + a
                    j = 2 * Xc + b
                    k = 2 * Yc + c
                    terms[(i, j, k)] += s

    # C11 = M1 + M2 + M3 - M4
    macro(0, 0, 0, 0, 0, 0, +1)  # M1 = A11*B11
    macro(0, 1, 1, 0, 0, 0, +1)  # M2 = A12*B21
    macro(1, 0, 0, 1, 0, 0, +1)  # M3 = A21*B12
    macro(1, 1, 1, 1, 0, 0, -1)  # -M4 = -(A22*B22)
    # C12 = M3 + M7
    macro(1, 0, 0, 1, 0, 1, +1)  # M3 = A21*B12
    macro(0, 0, 0, 1, 0, 1, +1)  # M7 part: A11*B12
    macro(0, 0, 1, 1, 0, 1, +1)  # M7 part: A11*B22
    macro(1, 0, 0, 1, 0, 1, -1)  # M7 part: -A21*B12
    macro(1, 0, 1, 1, 0, 1, -1)  # M7 part: -A21*B22
    # C21 = M6 + M4
    macro(0, 1, 1, 0, 1, 0, +1)  # M6: A12*B21
    macro(0, 1, 0, 0, 1, 0, -1)  # M6: -A12*B11
    macro(1, 1, 1, 0, 1, 0, +1)  # M6: A22*B21
    macro(1, 1, 0, 0, 1, 0, -1)  # M6: -A22*B11
    macro(1, 1, 1, 1, 1, 0, +1)  # M4 = A22*B22
    # C22 = M1 - M6 + M4 + M7
    macro(0, 0, 0, 0, 1, 1, +1)  # M1 = A11*B11
    macro(0, 1, 1, 0, 1, 1, -1)  # -M6: -A12*B21
    macro(0, 1, 0, 0, 1, 1, +1)  # -M6: +A12*B11
    macro(1, 1, 1, 0, 1, 1, -1)  # -M6: -A22*B21
    macro(1, 1, 0, 0, 1, 1, +1)  # -M6: +A22*B11
    macro(1, 1, 1, 1, 1, 1, +1)  # M4 = A22*B22
    macro(0, 0, 0, 1, 1, 1, +1)  # M7: A11*B12
    macro(0, 0, 1, 1, 1, 1, +1)  # M7: A11*B22
    macro(1, 0, 0, 1, 1, 1, -1)  # M7: -A21*B12
    macro(1, 0, 1, 1, 1, 1, -1)  # M7: -A21*B22

    keys = [key for key, v in terms.items() if v != 0]
    rank = len(keys)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    for r, (i, j, k) in enumerate(keys):
        U[i * 4 + j][r] = terms[(i, j, k)]
        V[j * 4 + k][r] = 1
        W[k * 4 + i][r] = 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 120.0
    cand = _block_strassen()
    if time.time() < deadline - 10.0 and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    return _schoolbook()

# EVOLVE-BLOCK-END
