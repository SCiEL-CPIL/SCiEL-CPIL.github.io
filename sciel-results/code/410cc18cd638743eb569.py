"""Fresh design: parametric shape family + L-BFGS-B over scale-invariant shape space."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
T_START = time.time()
TIME_LIMIT = 110.0  # safety margin under 120s; keep explicit guard

def _time_left():
    return TIME_LIMIT - (time.time() - T_START)

def _eval(h):
    """Compute the task's value for a nonnegative height array h (len N)."""
    h = np.asarray(h, dtype=float)
    if h.size != N or not np.all(np.isfinite(h)) or np.any(h < 0):
        return -1.0
    g = np.convolve(h, h)
    m = g.size
    y = np.concatenate(([0.0], g, [0.0]))
    xs = np.linspace(-0.5, 0.5, m + 2)
    w = xs[1] - xs[0]
    # l2 = sum_i (w/3)*(y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _gauss_shape(params):
    """Build a nonnegative height vector from a sum-of-Gaussians parametric shape.
    params: [amp1, mu1, sig1, amp2, mu2, sig2, base]
    mu in [-1,1] mapped to [0, N-1]; sig in (0.05, 4.0] via softplus-ish.
    Returns h (len N) >= 0.
    """
    amp1, mu1, sig1, amp2, mu2, sig2, base = params
    n = np.arange(N, dtype=float)
    # map mu from [-1,1] -> [0, N-1]
    c1 = (mu1 + 1.0) * 0.5 * (N - 1)
    c2 = (mu2 + 1.0) * 0.5 * (N - 1)
    s1 = 0.05 + np.exp(sig1)          # sig1 in R -> sig in (0.05, ...)
    s2 = 0.05 + np.exp(sig2)
    a1 = 1.0 + np.exp(amp1)           # positive amplitude
    a2 = 1.0 + np.exp(amp2)
    b0 = 0.0 + np.exp(base)           # nonnegative base
    h = b0 + a1 * np.exp(-0.5 * ((n - c1) / s1) ** 2) \
             + a2 * np.exp(-0.5 * ((n - c2) / s2) ** 2)
    return np.maximum(h, 0.0)

def _neg_value(params):
    h = _gauss_shape(params)
    return -_eval(h)

def construct():
    best_h = np.ones(N)
    best_val = _eval(best_h)

    rng = np.random.default_rng(0)

    # A diverse set of starting points in the 7-dim parametric space.
    # param layout: [amp1, mu1, sig1, amp2, mu2, sig2, base]
    def rand_params():
        amp1 = rng.uniform(-2.0, 2.0)
        mu1 = rng.uniform(-1.0, 1.0)
        sig1 = rng.uniform(np.log(0.3), np.log(3.0))
        amp2 = rng.uniform(-2.0, 2.0)
        mu2 = rng.uniform(-1.0, 1.0)
        sig2 = rng.uniform(np.log(0.3), np.log(3.0))
        base = rng.uniform(-3.0, 0.5)
        return np.array([amp1, mu1, sig1, amp2, mu2, sig2, base])

    def bounds():
        return [(-4.0, 4.0), (-1.0, 1.0), (np.log(0.05), np.log(4.0)),
                (-4.0, 4.0), (-1.0, 1.0), (np.log(0.05), np.log(4.0)),
                (-4.0, 1.0)]

    bnds = bounds()

    # Deterministic curated starts (single bump center, two bumps, broad, narrow)
    curated = [
        [0.0, 0.0, np.log(1.0), -1.0, 0.0, np.log(1.0), -3.0],   # one central bump
        [0.5, -0.5, np.log(0.8), 0.5, 0.5, np.log(0.8), -3.0],   # two symmetric bumps
        [0.0, 0.0, np.log(2.0), -1.0, 0.0, np.log(2.0), -3.0],   # broad bump
        [0.0, 0.0, np.log(0.4), -1.0, 0.0, np.log(0.4), -3.0],   # narrow bump
        [0.0, 0.0, np.log(1.0), -1.0, 0.0, np.log(1.0), -1.0],   # bump + base
    ]

    starts = [np.array(c, dtype=float) for c in curated]
    # add random starts
    for _ in range(10):
        starts.append(rand_params())

    for p0 in starts:
        if _time_left() < 8.0:
            break
        try:
            res = minimize(_neg_value, p0, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.x is not None:
                val = -res.fun
                if val > best_val + 1e-12:
                    best_val = val
                    best_h = _gauss_shape(res.x)
        except Exception:
            pass

    # If time remains, do a second pass with more random starts / more iterations.
    if _time_left() > 15.0:
        for _ in range(8):
            if _time_left() < 8.0:
                break
            p0 = rand_params()
            try:
                res = minimize(_neg_value, p0, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9})
                if res.x is not None:
                    val = -res.fun
                    if val > best_val + 1e-12:
                        best_val = val
                        best_h = _gauss_shape(res.x)
            except Exception:
                pass

    h = np.asarray(best_h, dtype=float)
    # ensure finite nonnegative
    h = np.where(np.isfinite(h), h, 0.0)
    h = np.maximum(h, 0.0)
    if not np.any(h > 0):
        h = np.ones(N)
    return list(h)
