"""Matrix multiplication 3x5x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
START = time.time()
DEADLINE = 150.0  # evaluator kills at 150s; keep margin
SAFETY = 10.0     # stop searching this many seconds before the limit


def _time_left():
    return (START + DEADLINE - SAFETY) - time.time()


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


# Strassen 7-term decomposition for a 2x2x2 block.
# Local A index = i*2+j, local B index = j*2+k, local C index = k*2+i.
_U2 = [
    [1, 0, 0, 1],   # M1: A00+A11
    [0, 0, 1, 1],   # M2: A10+A11
    [1, 0, 0, 0],   # M3: A00
    [0, 0, 0, 1],   # M4: A11
    [1, 1, 0, 0],   # M5: A00+A01
    [-1, 0, 1, 0],  # M6: A10-A00
    [0, 1, 0, -1],  # M7: A01-A11
]
_V2 = [
    [1, 0, 0, 1],   # B00+B11
    [1, 0, 0, 0],   # B00
    [0, 0, 1, -1],  # B10-B11
    [-1, 1, 0, 0],  # B01-B00
    [0, 0, 0, 1],   # B11
    [1, 0, 1, 0],   # B00+B10
    [0, 1, 0, 1],   # B01+B11
]
_W2 = [
    [1, 0, 0, 1],   # C00+1, C11+1
    [0, 0, 1, 0],   # C10+1
    [0, 1, 0, 1],   # C01+1, C11+1
    [1, 0, 1, 0],   # C00+1, C10+1
    [-1, 1, 0, 0],  # C00-1, C01+1
    [0, 0, 0, 1],   # C11+1
    [1, 0, 0, 0],   # C00+1
]


def _block_rank(a, b, c):
    if a == 2 and b == 2 and c == 2:
        return 7
    return a * b * c


def _tilings(n):
    """All ways to split n into parts of size 1 and 2 (and 3), in order."""
    parts = []
    def rec(rem, cur):
        if rem == 0:
            parts.append(list(cur))
            return
        for s in (2, 1, 3):
            if s <= rem:
                cur.append(s)
                rec(rem - s, cur)
                cur.pop()
    rec(n, [])
    return parts


def _build(t_i, t_j, t_k):
    n, m, p = DIMENSIONS
    # compute total rank
    rank = 0
    for a in t_i:
        for b in t_j:
            for c in t_k:
                rank += _block_rank(a, b, c)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    i0 = 0
    for a in t_i:
        j0 = 0
        for b in t_j:
            k0 = 0
            for c in t_k:
                if a == 2 and b == 2 and c == 2:
                    for t in range(7):
                        for i in range(2):
                            for j in range(2):
                                U[(i0 + i) * m + (j0 + j), col] += _U2[t][i * 2 + j]
                        for j in range(2):
                            for k in range(2):
                                V[(j0 + j) * p + (k0 + k), col] += _V2[t][j * 2 + k]
                        for k in range(2):
                            for i in range(2):
                                W[(k0 + k) * n + (i0 + i), col] += _W2[t][k * 2 + i]
                        col += 1
                else:
                    for i in range(a):
                        for j in range(b):
                            for k in range(c):
                                U[(i0 + i) * m + (j0 + j), col] = 1
                                V[(j0 + j) * p + (k0 + k), col] = 1
                                W[(k0 + k) * n + (i0 + i), col] = 1
                                col += 1
                k0 += c
            j0 += b
        i0 += a
    assert col == rank
    return U, V, W, rank


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    result = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.array_equal(result, T)


def construct():
    best = None
    best_rank = None
    for t_i in _tilings(3):
        if _time_left() <= 0:
            break
        for t_j in _tilings(5):
            if _time_left() <= 0:
                break
            for t_k in _tilings(7):
                if _time_left() <= 0:
                    break
                U, V, W, r = _build(t_i, t_j, t_k)
                if not _verify(U, V, W):
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best = (U, V, W)
    if best is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
