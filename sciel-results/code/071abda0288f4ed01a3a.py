import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0


def _value(h):
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_curr = y[:-1]
    y_next = y[1:]
    l2 = (w / 3.0) * np.sum(y_curr**2 + y_curr * y_next + y_next**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    return -_value(h)


def _params_to_h(params):
    """Convert params (A, c, sigma) for K Gaussians to a 50-dim height vector."""
    K = len(params) // 3
    h = np.zeros(N)
    x = np.linspace(0, 1, N)
    for k in range(K):
        A = params[3 * k]
        c = params[3 * k + 1]
        s = params[3 * k + 2]
        if s < 1e-6:
            s = 1e-6
        h += A * np.exp(-((x - c)**2) / (2.0 * s**2))
    return np.maximum(h, 0.0)


def _neg_value_params(params):
    try:
        h = _params_to_h(params)
        if not np.all(np.isfinite(h)):
            return 1e10
        return -_value(h)
    except Exception:
        return 1e10


def _concentrated(beta):
    return [beta**i for i in range(N)]


def _cosine_peak(c, amp, width):
    x = np.arange(N) / N
    d = np.minimum(np.abs(x - c), 1.0 - np.abs(x - c))
    return amp * np.cos(np.pi * d / width)**2


def _random_init(rng, kind):
    if kind == 0:
        return _concentrated(0.92)
    if kind == 1:
        return _concentrated(0.85)
    if kind == 2:
        return _concentrated(0.75)
    if kind == 3:
        return list(_cosine_peak(0.0, 1.0, 0.5))
    if kind == 4:
        return list(_cosine_peak(0.0, 1.0, 0.3))
    if kind == 5:
        return list(_cosine_peak(0.0, 1.0, 0.2))
    if kind == 6:
        return list(_cosine_peak(0.25, 1.0, 0.4))
    if kind == 7:
        return list(_cosine_peak(0.5, 1.0, 0.4))
    return list(rng.random(N) * 1.5)


def _coordinate_descent(h, deadline, iters=600):
    h = np.array(h, dtype=float)
    best = _value(h)
    step = 0.05
    n = len(h)
    it = 0
    while it < iters and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for direction in (1, -1):
                trial = h.copy()
                trial[i] = max(0.0, h[i] + direction * step)
                v = _value(trial)
                if v > best + 1e-12:
                    h = trial
                    best = v
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
        it += 1
    return h, best


def _nelder_mead(h, deadline):
    try:
        res = minimize(_neg_value, h, method='Nelder-Mead',
                       options={'maxiter': 4000, 'maxfev': 6000,
                                'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if time.time() < deadline and res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if _value(hx) > _value(h):
                return hx
    except Exception:
        pass
    return np.array(h, dtype=float)


def _lbfgs_polish(h, deadline):
    try:
        bounds_full = [(0, None)] * N
        res = minimize(
            _neg_value, h, method='L-BFGS-B',
            bounds=bounds_full,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0.0, None)
            if _value(h_polished) > _value(h):
                return h_polished
    except Exception:
        pass
    return np.array(h, dtype=float)


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    best_h = np.ones(N)
    best_v = _value(best_h)

    # Stage 1: Parametric Gaussian-mixture global search (from inspiration 1)
    # This reduces dimensionality to enable many random restarts for global exploration.
    K = 3
    dim = 3 * K
    bounds = [(0, 5), (0, 1), (0.01, 0.5)] * K
    n_restarts = 120

    for i in range(n_restarts):
        if time.time() > deadline - 30:  # leave time for local polish
            break

        params0 = np.zeros(dim)
        for k in range(K):
            params0[3 * k] = rng.uniform(0.1, 2.0)
            params0[3 * k + 1] = rng.uniform(0, 1)
            params0[3 * k + 2] = rng.uniform(0.05, 0.3)

        try:
            res = minimize(
                _neg_value_params, params0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                h_opt = _params_to_h(res.x)
                val_opt = _value(h_opt)
                if val_opt > best_v:
                    best_v = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Stage 2: Try analytic initializations with coordinate descent (from current program)
    for kind in range(8):
        if time.time() > deadline - 20:
            break
        h0 = _random_init(rng, kind)
        h, v = _coordinate_descent(h0, deadline - 15, iters=200)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # Stage 3: Coordinate descent from the best found
    if time.time() < deadline - 10:
        h, v = _coordinate_descent(best_h, deadline - 8, iters=500)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # Stage 4: L-BFGS-B polish (from inspiration 1)
    if time.time() < deadline - 5:
        h = _lbfgs_polish(best_h, deadline - 3)
        v = _value(h)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # Stage 5: Nelder-Mead polish (from current program)
    if time.time() < deadline - 3:
        h = _nelder_mead(best_h, deadline - 2)
        v = _value(h)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # Stage 6: Random perturbations + polish if time remains
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            pert = np.array(best_h, dtype=float)
            pert = np.clip(pert + rng.normal(0, 0.1, N), 0.0, None)
            h, v = _coordinate_descent(pert, deadline - 3, iters=200)
            if v > best_v:
                best_v, best_h = v, h.copy()

    # Final polish
    if time.time() < deadline - 1:
        h = _lbfgs_polish(best_h, deadline - 0.5)
        v = _value(h)
        if v > best_v:
            best_h = h.copy()

    best_h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)

    return [float(x) for x in best_h]
