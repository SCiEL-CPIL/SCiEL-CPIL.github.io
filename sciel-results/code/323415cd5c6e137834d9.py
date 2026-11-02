import time, random, math
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 138.0
_T0 = time.time()
_RNG = random.Random(1234567)

def _time_left():
    return DEADLINE - (time.time() - _T0)

def _snap(x, tol=1e-4):
    r = round(2.0 * x) / 2.0
    return r if abs(x - r) < tol else x

def _round_half(A):
    return (2.0 * np.asarray(A, dtype=float)).round() / 2.0

def _validate(U, V, W):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T = U @ V.T @ W.T
    ref = np.zeros((n * p, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[k * n + i, j * m + k] = 1.0
    if not np.allclose(T, ref, rtol=0.0, atol=1e-6):
        return False
    return bool(np.allclose(U, _round_half(U), rtol=0.0, atol=1e-4)) and \
           bool(np.allclose(V, _round_half(V), rtol=0.0, atol=1e-4)) and \
           bool(np.allclose(W, _round_half(W), rtol=0.0, atol=1e-4))

def _matmul(U, V, W):
    return U @ V.T @ W.T

def _loss(U, V, W, ref):
    return float(np.sum((_matmul(U, V, W) - ref) ** 2))

def _local_search(U, V, W, ref, target, budget):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    cur = _loss(U, V, W, ref)
    if cur <= target:
        return U, V, W
    t = 0.02
    step = 0
    best = (U.copy(), V.copy(), W.copy(), cur)
    while cur > target and step < budget and _time_left() > 3.0:
        M = _RNG.choice([0, 1, 2])
        A = (U, V, W)[M]
        idx = tuple(_RNG.randrange(d) for d in A.shape)
        old = A[idx]
        A[idx] = old + _RNG.gauss(0.0, t)
        nl = _loss(U, V, W, ref)
        if nl < cur:
            cur = nl
            if cur < best[3]:
                best = (U.copy(), V.copy(), W.copy(), cur)
            t = min(t * 1.03, 0.05)
        else:
            A[idx] = old
            t = max(t * 0.97, 0.004)
        step += 1
    return best[0], best[1], best[2]

def _deterministic_construct():
    n, m, p = DIMENSIONS
    rank = n * m * p - 1
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    # Drop the last basis column and compensate by adding its factor to the first column.
    U[:, 0] += U[:, rank - 1]
    V[:, 0] += V[:, rank - 1]
    W[:, 0] += W[:, rank - 1]
    return U[:, :rank], V[:, :rank], W[:, :rank]

def _ref_tensor():
    n, m, p = DIMENSIONS
    ref = np.zeros((n * p, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[k * n + i, j * m + k] = 1.0
    return ref

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    ref = _ref_tensor()
    best = None

    # 1) Deterministic near-optimal construction (target rank 23).
    try:
        U0, V0, W0 = _deterministic_construct()
        U0, V0, W0 = _local_search(U0, V0, W0, ref, 1e-8, 4000)
        U0, V0, W0 = _round_half(U0), _round_half(V0), _round_half(W0)
        if _validate(U0, V0, W0):
            best = (U0, V0, W0)
    except Exception:
        pass

    # 2) Randomized local search trying to reach the known optimal rank 19.
    target_rank = 19
    target = 1e-8
    if _time_left() > 6.0:
        for it in range(300):
            if _time_left() < 6.0:
                break
            U = np.zeros((n * m, target_rank)); V = np.zeros((m * p, target_rank)); W = np.zeros((p * n, target_rank))
            for c in range(target_rank):
                U[:, c] = 1.0
                V[:, c] = 1.0
                W[:, c] = 1.0
                U[:, c] += _RNG.uniform(-0.3, 0.3, n * m)
                V[:, c] += _RNG.uniform(-0.3, 0.3, m * p)
                W[:, c] += _RNG.uniform(-0.3, 0.3, p * n)
            U, V, W = _local_search(U, V, W, ref, target, 6000)
            U, V, W = _round_half(U), _round_half(V), _round_half(W)
            if _validate(U, V, W):
                best = (U, V, W)
                break
            if best is None:
                # keep best-effort fallback if we ever lose track
                pass

    # 3) Fallback: schoolbook (always valid).
    if best is None:
        rank = schoolbook_rank
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        best = (U, V, W)

    U, V, W = best
    U = _round_half(U); V = _round_half(V); W = _round_half(W)
    if not _validate(U, V, W):
        # last-resort guaranteed-valid schoolbook
        rank = schoolbook_rank
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
