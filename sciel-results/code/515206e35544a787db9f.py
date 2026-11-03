"""Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.

Strategy:
  1. Build the correct schoolbook decomposition (rank n*m*p = 54) as a guaranteed
     valid starting point.
  2. Use Alternating Least Squares (ALS) to search for a lower-rank exact
     decomposition.  For a fixed rank r we iterate: fix two factors, solve for the
     third via linear least squares, and check exact recovery of the tensor.
  3. As a fallback / safety net, also attempt a column-merging reduction on the
     schoolbook solution.
The entry point construct() returns {'U','V','W'} with integer/half-integer
entries under the trace convention W[k*n+i, r].
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[a,b,c] with a=i*m+j, b=j*p+k,
    c=k*n+i.  T[a,b,c] = 1 iff the indices are consistent, else 0."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _schoolbook(n, m, p):
    """Correct schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _reconstruct(U, V, W):
    """Return the tensor reconstructed from factor matrices (as a flat array)."""
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    return np.einsum('ar,br,cr->abc', U, V, W)


def _error(U, V, W, T):
    R = _reconstruct(U, V, W)
    return float(np.sum((R - T) ** 2))


def _als_decompose(T, r, max_iter=300, tol=1e-12, time_limit=None):
    """Try to find a rank-r decomposition of T via ALS.  Returns (U,V,W) on
    success or None.  T has shape (A,B,C)."""
    A, B, C = T.shape
    best_err = float('inf')
    for trial in range(20):
        if time_limit is not None and time.time() > time_limit:
            break
        rng = np.random.default_rng(1000 + r * 100 + trial)
        U = rng.standard_normal((A, r))
        V = rng.standard_normal((B, r))
        W = rng.standard_normal((C, r))
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                return None
            # --- update U: for each a, solve min ||U[a,:] @ (V*W) - T[a]||
            # M_U[b*C + c, r] = V[b,r]*W[c,r];  target = T[a].flatten()
            MW = (V[:, :, None] * W[None, :, :]).reshape(B * C, r)
            MWt = MW.T  # (r, B*C)
            for a in range(A):
                tgt = T[a].reshape(B * C)
                # solve U[a,:] such that U[a,:] @ MW = tgt  =>  MWt @ U[a,:] = tgt
                sol, *_ = np.linalg.lstsq(MWt, tgt, rcond=None)
                U[a, :] = sol
            # --- update V
            UW = (U[:, :, None] * W[None, :, :]).reshape(A * C, r)
            UWt = UW.T
            for b in range(B):
                tgt = T[:, b, :].reshape(A * C)
                sol, *_ = np.linalg.lstsq(UWt, tgt, rcond=None)
                V[b, :] = sol
            # --- update W
            UV = (U[:, :, None] * V[None, :, :]).reshape(A * B, r)
            UVt = UV.T
            for c in range(C):
                tgt = T[:, :, c].reshape(A * B)
                sol, *_ = np.linalg.lstsq(UVt, tgt, rcond=None)
                W[c, :] = sol
            err = _error(U, V, W, T)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < tol:
                return U, V, W
        if best_err < tol:
            return best
    return None


def _try_column_merge(U, V, W, T, time_limit=None):
    """Attempt to reduce rank by detecting duplicate/linearly-dependent columns.
    Returns (U,V,W) of lower rank if found, else None."""
    r = U.shape[1]
    # A column r is redundant if the 3 vectors (U[:,r], V[:,r], W[:,r]) are
    # such that the rank-1 tensor they form is a linear combination of others.
    # Simple test: build the set of all rank-1 components and try dropping each.
    for c in range(r):
        if time_limit is not None and time.time() > time_limit:
            return None
        idx = [i for i in range(r) if i != c]
        U2, V2, W2 = U[:, idx], V[:, idx], W[:, idx]
        if _error(U2, V2, W2, T) < 1e-9:
            return U2, V2, W2
    return None


def construct():
    deadline = time.time() + 110.0  # safety margin under 120s cap
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Guaranteed valid schoolbook solution (rank 54).
    U, V, W = _schoolbook(n, m, p)
    assert _error(U, V, W, T) < 1e-9, "schoolbook failed"
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to beat schoolbook with ALS for decreasing ranks.
    # Lower bound on tensor rank for n x m x p is at least max(n*m, m*p, p*n)
    # and at least max(n, m, p) etc.  Known: rank >= max(n*m, m*p, p*n) is NOT
    # true in general; a safe lower bound is max(n, m, p).  We'll just scan down.
    lower = max(n, m, p)  # = 9
    for r in range(lower, best_rank):
        if time.time() > deadline - 30.0:
            break
        res = _als_decompose(T, r, max_iter=200, time_limit=deadline - 30.0)
        if res is not None:
            U, V, W = res
            if _error(U, V, W, T) < 1e-9:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # found the smallest r in this scan that works

    # Try column-merging reduction on the best solution found.
    if time.time() < deadline - 10.0:
        merged = _try_column_merge(best_U, best_V, best_W, T, time_limit=deadline - 10.0)
        if merged is not None:
            best_U, best_V, best_W = merged
            best_rank = best_U.shape[1]

    # Snap near-integer / half-integer values to exact rationals for cleanliness.
    def snap(M):
        M = np.array(M, dtype=np.float64)
        out = np.round(M * 2.0) / 2.0
        # If values are very close to integers, prefer integers.
        close_int = np.abs(out - np.round(out)) < 1e-6
        out = np.where(close_int, np.round(out), out)
        return out

    Uf, Vf, Wf = snap(best_U), snap(best_V), snap(best_W)
    # Verify the snapped solution is still exact.
    if _error(Uf, Vf, Wf, T) < 1e-9:
        best_U, best_V, best_W = Uf, Vf, Wf

    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
