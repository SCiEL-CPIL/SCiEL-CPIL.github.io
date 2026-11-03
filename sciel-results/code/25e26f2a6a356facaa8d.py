"""Erdos minimum overlap: optimize step heights to minimize max overlap with shifted complement."""
import numpy as np
from scipy.optimize import minimize

N = 100


def _overlaps(h):
    """Return vector of overlap integrals for each circular shift."""
    one_minus_h = 1.0 - h
    n = len(h)
    overlaps = np.empty(n)
    for k in range(n):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.sum(h * shifted)
    return overlaps


def _max_overlap(h):
    return np.max(_overlaps(h))


def _logsumexp_overlaps(h, beta=50.0):
    ov = _overlaps(h)
    m = np.max(ov)
    return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta


def _project(h):
    """Project h onto {0<=h<=1, sum=h.sum()} approximately by clipping and rescaling."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 0:
        h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    import time
    t0 = time.time()
    DEADLINE = 110.0  # safety margin before 120s

    # Start from a non-uniform initialization to escape the uniform plateau.
    # Use a block-like or sinusoidal pattern.
    rng = np.random.default_rng(42)
    x = np.linspace(0, 2, N, endpoint=False)
    # Try a few initializations
    inits = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * x / 2.0),
        0.5 + 0.3 * np.sin(4 * np.pi * x / 2.0),
        (x < 1.0).astype(float),  # block
        rng.uniform(0, 1, N),
    ]

    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    def time_left():
        return DEADLINE - (time.time() - t0)

    for init in inits:
        if time_left() < 10.0:
            break
        h0 = np.clip(init, 0, 1)
        # Rescale to sum N/2
        s = h0.sum()
        if s > 0:
            h0 = h0 * (N / 2.0) / s
            h0 = np.clip(h0, 0, 1)

        # Optimize log-sum-exp surrogate
        def obj(h):
            return _logsumexp_overlaps(h, beta=50.0)

        def obj_grad(h):
            # numerical gradient
            eps = 1e-6
            g = np.zeros_like(h)
            base = obj(h)
            for i in range(N):
                hp = h.copy(); hp[i] += eps
                hm = h.copy(); hm[i] -= eps
                g[i] = (obj(hp) - obj(hm)) / (2 * eps)
            return g

        res = minimize(obj, h0, method='L-BFGS-B',
                       bounds=[(0.0, 1.0)] * N,
                       options={'maxiter': 200, 'gtol': 1e-8})
        if time_left() < 5.0:
            break
        h_opt = _project(res.x)
        sc = _max_overlap(h_opt)
        if sc < best_score:
            best_score = sc
            best_h = h_opt

    # Polish: high-beta log-sum-exp + exact max
    for beta in [100.0, 200.0, 500.0]:
        if time_left() < 5.0:
            break
        def obj2(h):
            return _logsumexp_overlaps(h, beta=beta)
        res2 = minimize(obj2, best_h, method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 100, 'gtol': 1e-10})
        if time_left() < 3.0:
            break
        hp = _project(res2.x)
        sc = _max_overlap(hp)
        if sc < best_score:
            best_score = sc
            best_h = hp

    # Final projection to ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    s = best_h.sum()
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
