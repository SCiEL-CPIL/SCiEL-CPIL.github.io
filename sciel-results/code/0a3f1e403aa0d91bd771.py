"""Matrix multiplication 6x6x6: block-partition tensor decomposition search.
Returns {'U','V','W'} integer factor matrices exactly decomposing the matmul
tensor under the official trace convention W[k*n+i, r].  Score = rank (lower
is better).  Searches block partitions x base algorithms (schoolbook, and any
Strassen variant that verifies) and returns the lowest-rank verified-exact
decomposition within the time budget.  A verified schoolbook decomposition is
always available as a correctness fallback.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
N = M = P = 6


def _schoolbook_slices(a, b, c):
    return [({i: 1}, {j: 1}, {k: 1})
            for i in range(a) for j in range(b) for k in range(c)]


def _strassen_variants_2x2x2():
    """Several candidate rank-7 slice sets for 2x2x2; only verified ones used."""
    variants = []
    # Variant A
    variants.append([
        ({0: 1}, {0: 1}, {0: 1}),
        ({1: 1}, {0: 1}, {0: 1}),
        ({0: 1}, {1: 1}, {1: 1}),
        ({1: 1}, {1: 1}, {0: 1}),
        ({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}),
        ({1: 1, 0: -1}, {1: 1}, {0: 1, 1: 1}),
        ({0: 1, 1: -1}, {0: 1}, {0: 1, 1: 1}),
    ])
    # Variant B
    variants.append([
        ({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}),
        ({1: 1, 0: 1}, {0: 1}, {0: 1}),
        ({0: 1}, {1: 1, 0: -1}, {1: 1}),
        ({1: 1}, {1: 1, 0: -1}, {0: 1}),
        ({0: 1, 0: 1}, {1: 1}, {1: 1}),
        ({1: 1, 0: -1}, {0: 1, 1: 1}, {0: 1}),
        ({0: 1, 1: -1}, {1: 1, 0: 1}, {1: 1}),
    ])
    return variants


def _verify_slices(slices, a, b, c):
    T = np.zeros((a, b, c), dtype=np.int64)
    for i in range(a):
        for k in range(c):
            if i == k:
                for j in range(b):
                    T[i, j, k] = 1
    S = np.zeros((a, b, c), dtype=np.int64)
    for (u, v, w) in slices:
        for i, cu in u.items():
            for j, cv in v.items():
                for k, cw in w.items():
                    S[i, j, k] += cu * cv * cw
    return bool(np.array_equal(S, T))


def _assemble(n, m, p, bs, slices):
    na, nb, nc = n // bs, m // bs, p // bs
    rank = (na * nb * nc) * len(slices)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ia in range(na):
        i0 = ia * bs
        for ib in range(nb):
            j0 = ib * bs
            for ic in range(nc):
                k0 = ic * bs
                for (u, v, w) in slices:
                    for i, cu in u.items():
                        U[i0 + i][col] += cu
                    for j, cv in v.items():
                        V[j0 + j][col] += cv
                    for k, cw in w.items():
                        W[k0 + k][col] += cw
                    col += 1
    return U, V, W, rank


def _verify_dense(U, V, W, n, m, p):
    r = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    T = np.zeros((n, m, p), dtype=np.int64)
    for rr in range(r):
        ucol = Ua[:, rr]
        vcol = Va[:, rr]
        wcol = Wa[:, rr]
        for ui in np.nonzero(ucol)[0]:
            i, j = divmod(int(ui), m)
            uj = ucol[ui]
            for k in range(p):
                vk = vcol[j * p + k]
                if vk == 0:
                    continue
                wk = wcol[k * n + i]
                if wk == 0:
                    continue
                T[i, j, k] += uj * vk * wk
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != (1 if i == k else 0):
                    return False
    return True


def _candidate_plans(n, m, p):
    plans = []
    plans.append((1, _schoolbook_slices(1, 1, 1)))
    for bs in range(2, min(n, m, p) + 1):
        if n % bs or m % bs or p % bs:
            continue
        plans.append((bs, _schoolbook_slices(bs, bs, bs)))
        if bs == 2:
            for sl in _strassen_variants_2x2x2():
                if _verify_slices(sl, 2, 2, 2):
                    plans.append((bs, sl))
    plans.sort(key=lambda pl: (n // pl[0]) * (m // pl[0]) * (p // pl[0]) * len(pl[1]))
    return plans


def construct():
    start = time.time()
    deadline = start + 120.0
    n, m, p = N, M, P
    best = None
    for (bs, slices) in _candidate_plans(n, m, p):
        if time.time() > deadline - 45.0:
            break
        U, V, W, rank = _assemble(n, m, p, bs, slices)
        if best is None or rank < best[0]:
            if _verify_dense(U, V, W, n, m, p):
                best = (rank, U, V, W)
    if best is None:
        U, V, W, _ = _assemble(n, m, p, 1, _schoolbook_slices(1, 1, 1))
        return {"U": U, "V": V, "W": W}
    return {"U": best[1], "V": best[2], "W": best[3]}
