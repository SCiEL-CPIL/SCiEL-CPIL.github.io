import time
import numpy as np

_N = 64
_M = 20000
_A = (np.arange(_N) + 1) / _N          # (64,)
_Y = (np.arange(_M) + 0.5) / _M        # (20000,)
_W = (1.0 - _Y) / _N                   # (20000,)
_DEADLINE = 630.0 - 30.0               # safety margin

# Fast version for optimization
_M_OPT = 1000
_IDX_OPT = np.linspace(0, _M - 1, _M_OPT).astype(int)
_Y_OPT = _Y[_IDX_OPT]
_W_OPT = _W[_IDX_OPT]


def _compute_area_fast(x):
    """Union area of the 64 triangles using reduced samples for speed."""
    L = x[None, :] + _A[None, :] * _Y_OPT[:, None]   # (M_OPT, N)
    Ls = np.sort(L, axis=1)                          # (M_OPT, N)
    diffs = Ls[:, 1:] - Ls[:, :-1]                   # (M_OPT, N-1)
    w_col = _W_OPT[:, None]                          # (M_OPT, 1)
    union = _W_OPT + np.sum(np.minimum(diffs, w_col), axis=1)
    return float(np.mean(union))


def _compute_area_full(x):
    """Union area of the 64 triangles (exact quadrature)."""
    L = x[None, :] + _A[None, :] * _Y[:, None]   # (M, N)
    Ls = np.sort(L, axis=1)                      # (M, N)
    diffs = Ls[:, 1:] - Ls[:, :-1]               # (M, N-1)
    w_col = _W[:, None]                          # (M, 1)
    union = _W + np.sum(np.minimum(diffs, w_col), axis=1)
    return float(np.mean(union))


def _polish(x, deadline):
    """Nelder-Mead polish with a fast objective; safe on failure/timeout."""
    try:
        from scipy.optimize import minimize
        res = minimize(
            _compute_area_fast,
            x,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6},
        )
        return np.asarray(res.x, dtype=float)
    except Exception:
        return x


def kakeya_needle_2d_positions64() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(0)

    # Seed candidate: the analytic "converge apexes" point (cheap, valid).
    best = -0.5 * _A.copy()
    best_area = _compute_area_fast(best)

    # Phase 1: multi-start random-restart basin search.
    n_starts = 50
    for _ in range(n_starts):
        if time.time() > _DEADLINE:
            break
        x0 = rng.normal(0.0, 0.5, _N)
        x0 = _polish(x0, _DEADLINE)
        if not np.all(np.isfinite(x0)):
            continue
        a = _compute_area_fast(x0)
        if a < best_area:
            best_area = a
            best = x0.copy()

    # Phase 2: targeted refinement around the best basin.
    for _ in range(20):
        if time.time() > _DEADLINE:
            break
        x0 = best + rng.normal(0.0, 0.1, _N)
        x0 = _polish(x0, _DEADLINE)
        if not np.all(np.isfinite(x0)):
            continue
        a = _compute_area_fast(x0)
        if a < best_area:
            best_area = a
            best = x0.copy()

    if not np.all(np.isfinite(best)):
        return np.zeros(_N)
    return best
