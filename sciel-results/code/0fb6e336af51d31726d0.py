# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt


def _shannon(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def _proj_entropy(P, r):
    n = P.shape[0]
    ys = np.arange(n)
    xs = np.arange(n)
    keys = xs[None, :] + r * ys[:, None]
    uniq, inv = np.unique(keys, return_inverse=True)
    agg = np.bincount(inv.ravel(), weights=P.ravel(), minlength=uniq.size)
    return _shannon(agg)


def _objective(P):
    D = _proj_entropy(P, -1.0)
    H0 = _proj_entropy(P, 0.0)
    H1 = _proj_entropy(P, 1.0)
    H2 = _proj_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return -1e18
    return D / M


def _softmax_max(H, k):
    H = np.asarray(H, dtype=np.float64)
    m = np.max(H)
    return m + np.log(np.sum(np.exp(k * (H - m)))) / k


def _neg_soft_ratio(P, k):
    D = _proj_entropy(P, -1.0)
    H0 = _proj_entropy(P, 0.0)
    H1 = _proj_entropy(P, 1.0)
    H2 = _proj_entropy(P, 2.0)
    M = _softmax_max(np.array([H0, H1, H2]), k)
    if M <= 1e-9:
        return 1e18
    return -D / M


def _project(u):
    P = np.clip(u, 0.0, None)
    s = P.sum()
    if s <= 0.0:
        return np.full((10, 10), 1.0 / 100.0)
    return P / s


def _neg_soft_ratio_flat(u, k):
    return _neg_soft_ratio(_project(u), k)


def _neg_ratio_flat(u):
    return -_objective(_project(u))


def _make_starts():
    n = 10
    starts = []
    starts.append(np.full((n, n), 1.0 / 100.0))
    # structured: emphasize anti-diagonal (x - y = const)
    for c in range(-9, 10):
        M = np.zeros((n, n))
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                M[y, x] = 1.0
        if M.sum() > 0:
            starts.append(M / M.sum())
    # random
    rng = np.random.default_rng(12345)
    for _ in range(12):
        M = rng.random((n, n))
        starts.append(M / M.sum())
    return starts


def _local_search(u0, deadline, k=6.0):
    n = 10
    u0 = np.asarray(u0, dtype=np.float64).ravel()
    u0 = _project(u0).ravel()
    best_P = _project(u0)
    best_val = _objective(best_P)
    res = opt.minimize(_neg_soft_ratio_flat, u0, args=(k,), method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-10})
    if time.time() > deadline:
        return best_P, best_val
    P = _project(res.x)
    v = _objective(P)
    if v > best_val:
        best_P, best_val = P, v
    # high-precision polish on exact ratio
    res2 = opt.minimize(_neg_ratio_flat, P.ravel(), method="L-BFGS-B",
                        options={"maxiter": 800, "ftol": 1e-16, "gtol": 1e-12})
    if time.time() > deadline:
        return best_P, best_val
    P2 = _project(res2.x)
    v2 = _objective(P2)
    if v2 > best_val:
        best_P, best_val = P2, v2
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    deadline = time.time() + 500.0
    starts = _make_starts()
    best_P = starts[0]
    best_val = _objective(best_P)
    for i, s in enumerate(starts):
        if time.time() > deadline:
            break
        P, v = _local_search(s, deadline)
        if v > best_val:
            best_val = v
            best_P = P
    P = np.clip(np.asarray(best_P, dtype=np.float64), 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.full((10, 10), 1.0 / 100.0)
    else:
        P = P / s
    return P


# EVOLVE-BLOCK-END
