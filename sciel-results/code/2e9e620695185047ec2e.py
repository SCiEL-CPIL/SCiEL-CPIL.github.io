import time
import numpy as np

_N = 64
_M = 20000
_A = (np.arange(_N) + 1) / _N          # (64,)
_Y = (np.arange(_M) + 0.5) / _M        # (20000,)
_W = (1.0 - _Y) / _N                   # (20000,)
_WB = _W[None, :]
_DEADLINE = 630.0 - 30.0               # safety margin


def _compute_area(x):
    """Union area of the 64 triangles (exact quadrature)."""
    L = x[None, :] + _A[None, :] * _Y[:, None]   # (M, N)
    Ls = np.sort(L, axis=1)
    diffs = Ls[1:, :] - Ls[:-1, :]               # (M, N-1)
    union = _W + np.sum(np.minimum(diffs, _WB), axis=1)
    return float(np.mean(union))


def _gradient(x):
    """Numerical gradient of area (central differences)."""
    g = np.empty(_N)
    h = 1e-5
    base = _compute_area(x)
    for i in range(_N):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_compute_area(xp) - _compute_area(xm)) / (2.0 * h)
    return g


def _polish(x, deadline):
    """L-BFGS-B polish with a numerical gradient; safe on failure/timeout."""
    try:
        from scipy.optimize import minimize
        res = minimize(
            _compute_area,
            x,
            jac=_gradient,
            method='L-BFGS-B',
            options={'maxiter': 400, 'maxfun': 8000, 'maxls': 200},
        )
        return np.asarray(res.x, dtype=float)
    except Exception:
        return x


def kakeya_needle_2d_positions64() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(0)

    # Seed candidate: the analytic "converge apexes" point (cheap, valid).
    best = -0.5 * _A.copy()
    best_area = _compute_area(best)

    # Phase 1: multi-start random-restart basin search.
    n_starts = 80
    for _ in range(n_starts):
        if time.time() > _DEADLINE:
            break
        x0 = rng.normal(0.0, 0.5, _N)
        x0 = _polish(x0, _DEADLINE)
        if not np.all(np.isfinite(x0)):
            continue
        a = _compute_area(x0)
        if a < best_area:
            best_area = a
            best = x0.copy()

    # Phase 2: targeted refinement around the best basin.
    for _ in range(12):
        if time.time() > _DEADLINE:
            break
        x0 = best + rng.normal(0.0, 0.1, _N)
        x0 = _polish(x0, _DEADLINE)
        if not np.all(np.isfinite(x0)):
            continue
        a = _compute_area(x0)
        if a < best_area:
            best_area = a
            best = x0.copy()

    if not np.all(np.isfinite(best)):
        return np.zeros(_N)
    return best
