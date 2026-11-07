"""Second autocorrelation inequality – optimize 50 nonnegative step heights."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
SEED = [1.0] * N
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0


def _eval_value(h):
    """Compute the task value for a given height vector h (length 50)."""
    h = np.asarray(h, dtype=float)
    # g = np.convolve(h, h) has length 2*N - 1 = 99
    g = np.convolve(h, h)
    m = len(g)  # 99
    # y = [0, g[0], ..., g[m-1], 0] -> length m+2 = 101
    y = np.zeros(m + 2)
    y[1:m+1] = g
    # points np.linspace(-0.5, 0.5, m+2) -> spacing w
    w = 1.0 / (m + 1)  # (0.5 - (-0.5)) / (m+1)
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _seed_value():
    return _eval_value(np.array(SEED))


def _negative_value(h):
    """Objective to minimize: negative of the task value."""
    return -_eval_value(h)


def _is_valid(h):
    h = np.asarray(h, dtype=float)
    return np.all(np.isfinite(h)) and np.all(h >= 0)


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - SAFETY_MARGIN

    best_h = np.array(SEED, dtype=float)
    best_val = _seed_value()

    # Phase 1: Nelder-Mead from seed
    try:
        res = minimize(_negative_value, best_h, method='Nelder-Mead',
                       options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        if _is_valid(res.x):
            v = _eval_value(res.x)
            if v > best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return list(best_h)

    # Phase 2: Try Nelder-Mead from a few perturbed starts
    rng = np.random.default_rng(42)
    for attempt in range(5):
        if time.time() > deadline:
            break
        # Perturb around best
        h0 = best_h + rng.normal(0, 0.1, size=N)
        h0 = np.clip(h0, 0, None)
        try:
            res = minimize(_negative_value, h0, method='Nelder-Mead',
                           options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if _is_valid(res.x):
                v = _eval_value(res.x)
                if v > best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return list(best_h)

    # Phase 3: Differential evolution (global) if time permits
    # Use a coarser bound to keep it fast
    bounds = [(0.0, 5.0)] * N
    try:
        # Use a limited number of iterations for DE
        res = differential_evolution(_negative_value, bounds, seed=42,
                                     maxiter=100, tol=1e-8,
                                     polish=False, init='latinhypercube',
                                     popsize=15)
        if _is_valid(res.x):
            v = _eval_value(res.x)
            if v > best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return list(best_h)

    # Phase 4: Final polish with tighter Nelder-Mead
    try:
        res = minimize(_negative_value, best_h, method='Nelder-Mead',
                       options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-14,
                                'adaptive': True})
        if _is_valid(res.x):
            v = _eval_value(res.x)
            if v > best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass

    # Ensure nonnegativity
    best_h = np.clip(best_h, 0, None)

    return list(best_h)
