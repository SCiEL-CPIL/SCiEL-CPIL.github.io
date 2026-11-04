"""Rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 12.0
_EPS = 1e-6
_VALUES = [0, 1, -1, 0.5, -0.5]
_VALSET = frozenset(_VALUES)
_RMAX = 27


def _check_halfint(x):
    return np.allclose(2.0 * x, np.round(2.0 * x), atol=1e-7)


def _verify(U, V, W):
    n = m = p = 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                tgt = 1.0 if i == k else 0.0
                if abs(s - tgt) > 1e-7:
                    return False
    return True


def _solve_rank(r):
    n = m = p = 3
    Nm = n * m  # 9
    Mpp = m * p  # 9
    Pn = p * n  # 9
    # index of (i,j,k) in the 27-vector
    idx = [[[] for _ in range(p)] for _ in range(m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx[j][k].append(i)
    # precompute the 9 index triples
    triples = []
    for j in range(m):
        for k in range(p):
            for i in idx[j][k]:
                triples.append((i, j, k))

    vals = np.array(_VALUES, dtype=float)
    # enumerate U row vectors: 9 rows x 5 values -> 5^9
    # We'll iterate and build U as (9, r)
    # For each U, build A (27, r) and solve for W (9, r)
    # A[t, c] = U[i*m+j, c]
    for u_tuple in itertools.product(range(5), repeat=9):
        if time.time() > _DEADLINE - _MARGIN:
            return None
        U = np.zeros((Nm, r))
        for row in range(Nm):
            U[row, :] = vals[list(u_tuple)]
        # Build A: for each t (i,j,k), A[t,c] = U[i*m+j,c]
        A = np.empty((27, r))
        for t, (i, j, k) in enumerate(triples):
            A[t, :] = U[i * m + j, :]
        # rhs[t] = 1 if i==k
        rhs = np.array([1.0 if i == k else 0.0 for (i, j, k) in triples])
        # Solve A W^T = rhs  ->  W (9, r)
        try:
            WT = np.linalg.solve(A, rhs)  # (r, 9)
        except np.linalg.LinAlgError:
            continue
        W = WT.T  # (9, r)
        if not _check_halfint(W):
            continue
        # Now solve for V: for each t (i,j,k): V[j*p+k,c] = T[i,j,k] / (U[i*m+j,c]*W[k*n+i,c])
        # This must be consistent across all t sharing the same V row index (j*p+k).
        V = np.zeros((Mpp, r))
        ok = True
        # For each V row index (j,p+k), gather constraints
        # V[j*p+k, c] must equal rhs[t] / (U[i*m+j,c] * W[k*n+i,c]) for all t with that (j,k)
        for j in range(m):
            for k in range(p):
                vrow = j * p + k
                found = False
                for t, (i, jj, kk) in enumerate(triples):
                    if jj != j or kk != k:
                        continue
                    uval = U[i * m + j, :]
                    wval = W[k * n + i, :]
                    denom = uval * wval
                    # For columns where denom == 0, rhs must be 0
                    if not np.allclose(denom * 0, 0, atol=1e-9):
                        pass
                    # Compute candidate V row where denom != 0
                    mask = np.abs(denom) > 1e-9
                    if np.any(mask):
                        cand = np.zeros(r)
                        cand[mask] = rhs[t] / denom[mask]
                        # where denom == 0, require rhs[t] == 0
                        if not np.allclose(rhs[t], 0.0, atol=1e-9):
                            # denom has zeros but rhs != 0 -> impossible
                            # actually denom is per-column; check per column
                            bad = (~mask) & (np.abs(rhs[t]) > 1e-9)
                            if np.any(bad):
                                ok = False
                                break
                        if found:
                            if not np.allclose(cand, V[vrow, :], atol=1e-7):
                                ok = False
                                break
                        else:
                            V[vrow, :] = cand
                            found = True
                    else:
                        # all denom zero -> require rhs==0
                        if not np.allclose(rhs[t], 0.0, atol=1e-9):
                            ok = False
                            break
                        if found:
                            # V row already set; but all denom zero here, no constraint
                            pass
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        if not _check_halfint(V):
            continue
        # Full verification
        if _verify(U, V, W):
            return (U, V, W)
    return None


def construct():
    best = None
    # start from known-good schoolbook rank 27 as fallback
    n = m = p = 3
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
    best = {"U": U, "V": V, "W": W}

    # Try to find lower ranks
    for r in range(1, _RMAX):
        if time.time() > _DEADLINE - _MARGIN:
            break
        res = _solve_rank(r)
        if res is not None:
            U, V, W = res
            best = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
            }
            # found a valid rank r; could keep searching lower but r is increasing
            # we want MINIMAL, so stop at first found (lowest r)
            break
    return best
# EVOLVE-BLOCK-END
