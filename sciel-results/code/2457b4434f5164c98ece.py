"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition.

Combines:
  * The proven explicit rank-44 structure (Strassen rank-7 per 2x2x2 block
    plus 4 rank-1 terms for the third A-column / third B-row) as a fast,
    always-valid fallback and warm start.
  * A time-bounded search (block-structured warm start + per-mode SVD
    refinement on a tensor-norm objective + high-precision polish) that can
    find a lower exact rank if one exists and is reachable.

The objective is the exact tensor-norm error (sum of squared errors over all
entries), which is smooth and well-defined even when the exact support is
unknown.  A solution is "exact" when this norm is ~0, which forces the
decomposition to equal the multiplication tensor on the 48 target entries and
to be zero everywhere else.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _build_rank44_decomposition():
    """Explicit rank-44 decomposition (Strassen per 2x2x2 block + rank-4 tail)."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        # Strassen rank-7 for A[0:2,0:2] * B[0:2,k0:k1+1]
        U[0, col] += 1; U[4, col] += 1; V[k0, col] += 1; V[8 + k1, col] += 1
        W[2 * k0, col] += 1; W[2 * k1 + 1, col] += 1; col += 1
        U[3, col] += 1; U[4, col] += 1; V[k0, col] += 1
        W[2 * k0 + 1, col] += 1; W[2 * k1 + 1, col] += -1; col += 1
        U[0, col] += 1; V[k1, col] += 1; V[8 + k1, col] += -1
        W[2 * k1, col] += 1; W[2 * k1 + 1, col] += 1; col += 1
        U[4, col] += 1; V[8 + k0, col] += 1; V[k0, col] += -1
        W[2 * k0, col] += 1; W[2 * k0 + 1, col] += 1; col += 1
        U[0, col] += 1; U[1, col] += 1; V[8 + k1, col] += 1
        W[2 * k0, col] += -1; W[2 * k1, col] += 1; col += 1
        U[3, col] += 1; U[0, col] += -1; V[k0, col] += 1; V[k1, col] += 1
        W[2 * k1 + 1, col] += 1; col += 1
        U[1, col] += 1; U[4, col] += -1; V[8 + k0, col] += 1; V[8 + k1, col] += 1
        W[2 * k0, col] += 1; col += 1
        # Rank-4 tail: A[0:2,2] * B[2,k0:k1+1]
        for ii in range(2):
            for kk in range(2):
                kv = k0 if kk == 0 else k1
                U[ii * 3 + 2, col] = 1
                V[16 + kv, col] = 1
                W[kv * 2 + ii, col] = 1
                col += 1
    assert col == 44, col
    return U, V, W


def _verify(U, V, W, tol=1e-7):
    """Return True iff U,V,W exactly decompose the 2x3x8 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                e = abs(T[a, b, c] - 1.0)
                if e > max_err:
                    max_err = e
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                for a2 in range(n * m):
                    if a2 == a:
                        continue
                    for b2 in range(m * p):
                        if b2 == b:
                            continue
                        for c2 in range(p * n):
                            if c2 == c:
                                continue
                            e = abs(T[a2, b2, c2])
                            if e > max_err:
                                max_err = e
    return max_err <= tol


def _norm(U, V, W, target, target_mask):
    """Exact tensor-norm error (sum of squared errors) vs the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    s = 0.0
    for a, b, c in target:
        s += (T[a, b, c] - 1.0) ** 2
    s += np.sum(T[~target_mask] ** 2)
    return s


def _refine(U, V, W, target, target_mask, deadline, tol=1e-11, max_iters=200):
    """Per-mode least-squares / SVD refinement driving the tensor norm to ~0."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        # ---- update U ----
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, *_ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except Exception:
                    pass
        # ---- update V ----
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n * p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        M[idx, :] = U[i * m + j, :] * W[kk * n + i, :]
                        idx += 1
                try:
                    x, *_ = np.linalg.lstsq(M, np.ones(n * p), rcond=None)
                    V[b, :] = x
                except Exception:
                    pass
        # ---- update W ----
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n * m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        M[idx, :] = U[ii * m + j, :] * V[j * p + k, :]
                        idx += 1
                try:
                    x, *_ = np.linalg.lstsq(M, np.ones(n * m), rcond=None)
                    W[c, :] = x
                except Exception:
                    pass
        err = _norm(U, V, W, target, target_mask)
        if err < tol:
            break
    if _norm(U, V, W, target, target_mask) < 1e-8:
        return U, V, W
    return None


def _round_and_verify(U, V, W, tol=1e-6):
    """Round to half-integers/ints and verify; return rounded lists or None."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for scale in (2, 1):
        Ur = np.round(Ua * scale) / scale
        Vr = np.round(Va * scale) / scale
        Wr = np.round(Wa * scale) / scale
        if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), tol=tol):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None


def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank-48 (always valid, exact).
    fb_rank = n * m * p
    U_fb = np.zeros((n * m, fb_rank))
    V_fb = np.zeros((m * p, fb_rank))
    W_fb = np.zeros((p * n, fb_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    # Fast, always-valid rank-44 explicit decomposition (fallback + warm start).
    U44, V44, W44 = _build_rank44_decomposition()
    if _verify(U44.tolist(), V44.tolist(), W44.tolist(), tol=1e-7):
        best = (U44.tolist(), V44.tolist(), W44.tolist(), 44)
    else:
        best = (U_fb.tolist(), V_fb.tolist(), W_fb.tolist(), fb_rank)

    start = time.time()
    deadline = start + 120.0  # generous safety margin under the 150 s kill

    # Target support of the multiplication tensor.
    target = []
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target.append((a, b, c))
                target_mask[a, b, c] = True

    # ---- Search for a lower exact rank, bounded by the deadline ----
    for rank in range(40, best[3]):
        if time.time() > deadline - 5.0:
            break
        found = False
        # Warm start: block-structured (drop one tail term per block) then refine.
        try:
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            c = 0
            for b in range(4):
                k0 = 2 * b
                k1 = 2 * b + 1
                # 7 Strassen terms
                U[0, c] += 1; U[4, c] += 1; V[k0, c] += 1; V[8 + k1, c] += 1
                W[2 * k0, c] += 1; W[2 * k1 + 1, c] += 1; c += 1
                U[3, c] += 1; U[4, c] += 1; V[k0, c] += 1
                W[2 * k0 + 1, c] += 1; W[2 * k1 + 1, c] += -1; c += 1
                U[0, c] += 1; V[k1, c] += 1; V[8 + k1, c] += -1
                W[2 * k1, c] += 1; W[2 * k1 + 1, c] += 1; c += 1
                U[4, c] += 1; V[8 + k0, c] += 1; V[k0, c] += -1
                W[2 * k0, c] += 1; W[2 * k0 + 1, c] += 1; c += 1
                U[0, c] += 1; U[1, c] += 1; V[8 + k1, c] += 1
                W[2 * k0, c] += -1; W[2 * k1, c] += 1; c += 1
                U[3, c] += 1; U[0, c] += -1; V[k0, c] += 1; V[k1, c] += 1
                W[2 * k1 + 1, c] += 1; c += 1
                U[1, c] += 1; U[4, c] += -1; V[8 + k0, c] += 1; V[8 + k1, c] += 1
                W[2 * k0, c] += 1; c += 1
                # 3 tail terms (drop one of the four per block -> 10/block = 40)
                for (ii, kk) in [(0, 0), (0, 1), (1, 0)]:
                    kv = k0 if kk == 0 else k1
                    U[ii * 3 + 2, c] = 1
                    V[16 + kv, c] = 1
                    W[kv * 2 + ii, c] = 1
                    c += 1
            res = _refine(U, V, W, target, target_mask, deadline, tol=1e-11, max_iters=300)
            if res is not None:
                U, V, W = res
                if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-7):
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best = (rounded[0], rounded[1], rounded[2], rank)
                    else:
                        best = (U.tolist(), V.tolist(), W.tolist(), rank)
                    found = True
                    break
        except Exception:
            pass
        if found:
            break

        # Random restarts.
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            np.random.seed(1000 * rank + 7 * trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            try:
                res = _refine(U, V, W, target, target_mask, deadline, tol=1e-11, max_iters=250)
            except Exception:
                res = None
            if res is not None:
                U, V, W = res
                if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-7):
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best = (rounded[0], rounded[1], rounded[2], rank)
                    else:
                        best = (U.tolist(), V.tolist(), W.tolist(), rank)
                    found = True
                    break
        if found:
            break

    # Final polish on the best solution found (tighten to machine precision).
    if time.time() < deadline - 8.0:
        try:
            res = _refine(best[0], best[1], best[2], target, target_mask,
                          deadline, tol=1e-12, max_iters=400)
            if res is not None:
                U, V, W = res
                if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-7):
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best = (rounded[0], rounded[1], rounded[2], best[3])
                    else:
                        best = (U.tolist(), V.tolist(), W.tolist(), best[3])
        except Exception:
            pass

    return {"U": best[0], "V": best[1], "W": best[2]}
