# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize

_N = 20
_TIME_LIMIT = 630.0
_DEADLINE = time.time() + _TIME_LIMIT - 20.0  # safety margin before the 630s kill

# Precompute the index window for the "four distances" rule per k (0..18).
# k = argmin of the first gap; window is [k-1, k+2] clipped to [0, 18].
_WINDOWS = [list(range(max(0, k - 1), min(_N - 1, k + 3))) for k in range(_N - 1)]


def _radial_to_points(theta, a, b):
    """Map angular positions theta and radial Fourier coeffs (a: cos, b: sin) to (N,2)."""
    r = 1.0
    for m in range(1, 5):
        r = r + a[m] * np.cos(m * theta) + b[m] * np.sin(m * theta)
    r = np.clip(r, 0.4, 2.0)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _normalize(p):
    c = p.mean(axis=0)
    q = p - c
    d = np.sqrt(np.sum(q * q, axis=1))
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None
    return q / mean_d


def _vertex_score(p, i):
    """Exact per-vertex score from the contract (0.0 if any hard fail)."""
    d = np.sqrt(np.sum((p - p[i]) ** 2, axis=1))
    d = np.sort(np.delete(d, i))
    if np.min(d) < 0.05:
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    idx = _WINDOWS[k]
    four = d[idx]
    raw = np.mean(np.maximum(four / D, D / four))
    if raw <= 0.0:
        return 0.0
    return min(1.0, 1.0 / raw)


def _min_score(p):
    if p is None:
        return 0.0
    return min(_vertex_score(p, i) for i in range(_N))


def _soft_min(scores, beta):
    s = np.asarray(scores, dtype=float)
    m = np.min(s)
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (s - m))))


def _smooth_objective(theta, a, b, beta):
    """Negative soft-min of the per-vertex scores; returns +inf on invalid geometry."""
    p = _radial_to_points(theta, a, b)
    if np.any(~np.isfinite(p)):
        return 1e9
    # strict convexity: all cross products of consecutive edges share a sign
    e1 = p[1:] - p[:-1]
    e2 = p - p[1:]
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.min(np.abs(cross)) < 1e-5:
        return 1e9
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 1e9
    # no repeated points
    if np.min(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2).diagonal(offset=1)) < 1e-4:
        return 1e9
    q = _normalize(p)
    if q is None:
        return 1e9
    scores = np.array([_vertex_score(q, i) for i in range(_N)])
    return -_soft_min(scores, beta)


def _flatten(theta, a, b):
    return np.concatenate([theta, a, b])


def _unflatten(x):
    theta = x[0:_N]
    a = x[_N:_N + 4]
    b = x[_N + 4:_N + 8]
    return theta, a, b


def _regular_init():
    theta = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    a = np.zeros(4)
    b = np.zeros(4)
    return _flatten(theta, a, b)


def _jittered_init(rng, scale):
    theta = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    # small random radial Fourier perturbation
    a = rng.normal(0.0, scale, size=4)
    b = rng.normal(0.0, scale, size=4)
    return _flatten(theta, a, b)


def equidistant_points_polygon20() -> np.ndarray:
    best_x = _regular_init()
    best_val = _smooth_objective(*_unflatten(best_x), beta=20.0)
    if not np.isfinite(best_val):
        best_val = 1e9

    rng = np.random.default_rng(12345)
    beta = 20.0
    n_restarts = 14
    for r in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        if r == 0:
            x0 = _regular_init()
        else:
            x0 = _jittered_init(rng, scale=0.05 + 0.03 * (r % 3))

        x0 = x0.copy()
        # Nelder-Mead on the smooth surrogate
        res = minimize(
            lambda x: _smooth_objective(*_unflatten(x), beta=beta),
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": 1200,
                "maxfev": 3000,
                "xatol": 1e-6,
                "fatol": 1e-7,
                "adaptive": True,
            },
        )
        if time.time() > _DEADLINE:
            break
        val = res.fun
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_x = res.x.copy()

        # local polish with higher beta on the best so far
        if time.time() > _DEADLINE:
            break
        res2 = minimize(
            lambda x: _smooth_objective(*_unflatten(x), beta=beta * 3.0),
            best_x.copy(),
            method="Nelder-Mead",
            options={
                "maxiter": 800,
                "maxfev": 2000,
                "xatol": 1e-7,
                "fatol": 1e-8,
                "adaptive": True,
            },
        )
        if time.time() > _DEADLINE:
            break
        val2 = res2.fun
        if np.isfinite(val2) and val2 < best_val:
            best_val = val2
            best_x = res2.x.copy()

    # Final exact evaluation of the best candidate; fall back to regular if invalid.
    theta, a, b = _unflatten(best_x)
    p = _radial_to_points(theta, a, b)
    q = _normalize(p)
    if q is None or _min_score(q) <= 0.0:
        angles = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
        q = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return q


# EVOLVE-BLOCK-END
