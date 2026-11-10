"""Shared degree-five polynomial closure for the two-scale Lorenz-96 unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _design(x):
    x = np.asarray(x, dtype=np.float64)
    d = np.empty((x.size, 6), dtype=np.float64)
    d[:, 0] = 1.0
    for i in range(1, 6):
        d[:, i] = d[:, i - 1] * x
    return d


def _ols(D, y, lam):
    A = D.T @ D + lam * np.eye(6)
    b = D.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _irles(D, y, c0, iters=4, eps=1e-12):
    c = c0
    for _ in range(iters):
        r = y - D @ c
        s = np.sqrt(np.mean(r * r) + eps)
        w = np.minimum(1.0, (1.5 * s / (np.abs(r) + eps)) ** 2)
        W = np.diag(w)
        A = D.T @ W @ D + eps * np.eye(6)
        b = D.T @ (w * y)
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _fin(c):
    return np.asarray(c, dtype=np.float64).reshape(6)


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    DEADLINE = 120.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = x.size
    if n < 6:
        return np.zeros(6, dtype=np.float64)

    D = _design(x)

    # ---- 1) candidate pool: ridge sweep on the raw (uncentered) basis ----
    alphas = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    cands = {}
    for a in alphas:
        cands[a] = _ols(D, y, a)

    # ---- 2) pick alpha via a training holdout (proxy for validation) ----
    g = np.random.default_rng(0)
    idx = g.permutation(n)
    nval = max(1, int(round(0.15 * n)))
    tr, va = idx[nval:], idx[:nval]

    best_va, best_c = np.inf, None
    for a in alphas:
        if time.time() > DEADLINE - 15:
            break
        c = cands[a]
        err = float(np.mean((D[va] @ c - y[va]) ** 2))
        if err < best_va:
            best_va, best_c = err, c
    best_a = None

    # ---- 3) robust (IRLS) refit from a few candidate scales ----
    if time.time() < DEADLINE - 20:
        for a in [1e-2, 1e-1, 1.0, 10.0]:
            if time.time() > DEADLINE - 15:
                break
            c = _irles(D, y, cands[a], iters=4)
            err = float(np.mean((D[va] @ c - y[va]) ** 2))
            if err < best_va:
                best_va, best_c = err, c

    # ---- 4) sanity fallback: if best blew up, use the most regularized ----
    if best_c is None or not np.all(np.isfinite(best_c)) or np.max(np.abs(best_c)) > 1e6:
        best_c = cands[max(alphas)]

    if time.time() > DEADLINE - 5:
        return _fin(best_c)

    # ---- 5) final polish: tiny ridge on the selected scale ----
    base = best_c
    for a in [0.0, 1e-4, 1e-3, 1e-2]:
        if time.time() > DEADLINE - 5:
            break
        c = _ols(D, y, a)
        if np.all(np.isfinite(c)) and np.max(np.abs(c)) <= 1e6:
            base = c
            break

    return _fin(base)
# EVOLVE-BLOCK-START
