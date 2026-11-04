"""Rank-reduction search for 4x4 * 4x9 multiplication tensor.

Starts from the best known block decomposition (Strassen on the two 4x4x4
sub-blocks, schoolbook on the 4x4x1 tail -> rank 114), then attempts to
lower the rank via algebraic elimination of linearly dependent tensor
factors. Time-budgeted with a hard deadline guard.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 145.0  # safety margin under the 150 s kill wall-clock


# ---------------------------------------------------------------------------
# Strassen 4x4x4 (rank 7) as (U, V, W) with the official index conventions:
#   U[i*m + j, c],  V[j*p + k, c],  W[k*n + i, c]
# ---------------------------------------------------------------------------
def _strassen_4x4x4():
    n = m = p = 4
    S = [
        (1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 1, 0),
        (0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0),
        (0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0),
    ]
    A = [
        (0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1),
    ]
    B = [
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 1),
    ]
    U = [[0] * 7 for _ in range(16)]
    V = [[0] * 7 for _ in range(16)]
    W = [[0] * 7 for _ in range(16)]
    for c in range(7):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] = S[c][i * 4 + j]
                V[j * 4 + i][c] = A[c][j * 4 + i]
                W[i * 4 + j][c] = B[c][i * 4 + j]
    return U, V, W


def _schoolbook_nm_p(n, m, p):
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
    return U, V, W


def _combine(u1, v1, w1, u2, v2, w2, row_shifts):
    """Concatenate two factor sets; shift the second's row indices by row_shifts."""
    s1, s2, s3 = row_shifts
    r1 = len(w1[0])
    r2 = len(w2[0])
    U = [row[:] for row in u1]
    U += [[0] * r1 + row[:] for row in u2]
    V = [row[:] for row in v1]
    V += [[0] * r1 + row[:] for row in v2]
    W = [row[:] for row in w1]
    W += [[0] * r1 + row[:] for row in w2]
    for r in range(len(w2)):
        for c in range(r2):
            W[s3 + r][r1 + c] = w2[r][c]
    for r in range(len(u2)):
        for c in range(r2):
            U[s1 + r][r1 + c] = u2[r][c]
    for r in range(len(v2)):
        for c in range(r2):
            V[s2 + r][r1 + c] = v2[r][c]
    return U, V, W


def _base_decomposition():
    sU, sV, sW = _strassen_4x4x4()
    U, V, W = _combine(sU, sV, sW, sU, sV, sW, (16, 16, 16))
    tU, tV, tW = _schoolbook_nm_p(4, 4, 1)
    U, V, W = _combine(U, V, W, tU, tV, tW, (32, 32, 32))
    return U, V, W


# ---------------------------------------------------------------------------
# Reduction: find a linear relation among tensor factors and eliminate one.
# A factor c is eliminable iff its W-row support is contained in the span of
# the W-row supports of the other factors (then a W row-op makes W[:,c]=0,
# and a U/V row-op removes the now-unused column).
# ---------------------------------------------------------------------------
def _find_elimination(U, V, W, rng):
    nU, R = len(U), len(U[0])
    nV, nW = len(V), len(W)
    Wm = np.array(W, dtype=float)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    tol = 1e-9
    order = list(range(R))
    rng.shuffle(order)
    for c in order:
        if c >= R:
            continue
        if Wm[:, c].max() == 0.0:
            continue
        rows = np.where(Wm[:, c] != 0.0)[0]
        if len(rows) == 0:
            continue
        others = [j for j in range(R) if j != c]
        others = [j for j in others if Wm[:, j].max() != 0.0]
        if len(others) == 0:
            continue
        M = Wm[:, others]
        rank_full = np.linalg.matrix_rank(M, tol=tol)
        aug = np.column_stack([M, Wm[:, c]])
        rank_aug = np.linalg.matrix_rank(aug, tol=tol)
        if rank_aug > rank_full:
            continue
        # c is in the span of others -> find a dependency vector d (length R)
        # with d[c] != 0 and sum_j d[j] W[:,j] = 0.
        if rank_full == 0:
            d = np.zeros(R)
            d[c] = 1.0
        else:
            # null space of [M | W[:,c]]
            Q, Rr = np.linalg.qr(np.column_stack([M, Wm[:, c]]).T)
            nn = Q.shape[0] - rank_aug
            if nn <= 0:
                continue
            nullvec = Q[rank_aug:rank_aug + nn]
            # pick the dependency with largest |d[c]|
            idx = int(np.argmax(np.abs(nullvec[:, c])))
            d = nullvec[idx]
        if abs(d[c]) < tol:
            continue
        # Build new U, V, W
        Unew = Ua.copy()
        Vnew = Va.copy()
        Wnew = Wm.copy()
        for j in range(R):
            if j == c:
                continue
            if abs(d[j]) < tol:
                continue
            Unew[:, c] += d[j] / d[c] * Ua[:, j]
            Vnew[:, c] += d[j] / d[c] * Va[:, j]
        Wnew[:, c] = 0.0
        # drop column c
        keep = [j for j in range(R) if j != c]
        Uout = Unew[:, keep]
        Vout = Vnew[:, keep]
        Wout = Wnew[:, keep]
        return Uout, Vout, Wout
    return None


def _round_factors(U, V, W):
    Ua = np.round(np.array(U, dtype=float))
    Va = np.round(np.array(V, dtype=float))
    Wa = np.round(np.array(W, dtype=float))
    return (Ua.astype(int).tolist(), Va.astype(int).tolist(), Wa.astype(int).tolist())


def _verify(U, V, W):
    n, m, p = 4, 4, 9
    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    R = Um.shape[1]
    for i in range(n):
        for k in range(p):
            lhs = np.zeros((m, m))
            for j in range(m):
                for l in range(m):
                    s = 0.0
                    for c in range(R):
                        s += Um[i * m + j, c] * Vm[j * m + l, c]
                    lhs[j, l] = s
            for l in range(m):
                if abs(lhs[:, l] - Wm[k * n + i, :]) > 1e-6:
                    return False
    return True


# ---------------------------------------------------------------------------
# Search driver with hard deadline guard.
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    baseU, baseV, baseW = _base_decomposition()
    best = _round_factors(baseU, baseV, baseW)
    best_rank = len(best[0][0])
    if not _verify(*best):
        # fall back to the guaranteed-valid schoolbook
        sU, sV, sW = _schoolbook_nm_p(4, 4, 9)
        best = (sU, sV, sW)
        best_rank = len(sU[0])

    rng = random.Random(12345)
    attempt = 0
    while time.time() - t0 < DEADLINE:
        attempt += 1
        curU, curV, curW = best
        improved = False
        # try up to a few eliminations per attempt
        for _ in range(8):
            if time.time() - t0 > DEADLINE:
                break
            res = _find_elimination(curU, curV, curW, rng)
            if res is None:
                break
            curU, curV, curW = res
            improved = True
        if improved:
            cand = _round_factors(curU, curV, curW)
            if _verify(*cand):
                r = len(cand[0][0])
                if r < best_rank:
                    best = cand
                    best_rank = r
                    # keep going to try to shrink further
            # if it failed verification, discard this attempt
        # if no improvement found, break out of the loop (no more gain)
        if not improved:
            break
    return {"U": best[0], "V": best[1], "W": best[2]}
