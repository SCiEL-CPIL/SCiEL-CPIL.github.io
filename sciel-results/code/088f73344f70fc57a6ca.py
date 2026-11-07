"""Erdos minimum overlap: design a 100-step unit-integral height profile
that minimizes the max autocorrelation of (h, 1-h).

The binding constraint is the large-lag cross term sum_i (h[i] - h[i+k]),
maximized by a square wave (0 on one half, 1 on the other).  Smoothing the
transition (a short ramp) cuts that term, so the optimum is a smoothed step.
We search over a compact 3-parameter family (amplitude, phase, smoothness)
deterministically, then polish with Nelder-Mead under a wall-clock budget.
"""
import time

N = 100
_TOL = 1e-9 * N          # sum tolerance
_DEADLINE = time.time() + 135.0   # safety margin under the 150 s kill
import numpy as np

_IDX = np.arange(N, dtype=float)
_TWO_PI = 2.0 * np.pi


def _value(h):
    """Exact evaluator value (lower is better)."""
    return float(np.max(np.correlate(h, 1.0 - h, mode="full"))) / N * 2.0


def _sum_ok(h):
    return abs(float(h.sum()) - 50.0) <= _TOL


def _valid(h):
    if h.shape != (N,):
        return False
    if not np.all(np.isfinite(h)):
        return False
    if h.min() < -1e-9 or h.max() > 1.0 + 1e-9:
        return False
    return _sum_ok(h)


def _smooth_step(a, ph, w):
    """Smoothed square wave: 0.5 + a*sin(2*pi*(i+ph)/N) clipped to [0,1].
    a is the pre-clip amplitude (a<1 gives a rounded, smoothed transition)."""
    s = np.sin(_TWO_PI * (_IDX + ph) / N)
    return np.clip(0.5 + a * s, 0.0, 1.0)


def _smooth_step2(a, ph, w, period):
    """Generalized smooth step with a free period (for restarts)."""
    s = np.sin(_TWO_PI * (_IDX + ph) / period)
    return np.clip(0.5 + a * s, 0.0, 1.0)


def _golden(f, lo, hi, iters=40):
    """Minimize a scalar function on [lo, hi] via golden section."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
    xm = 0.5 * (lo + hi)
    return f(xm), xm


def _optimize_smooth_step():
    """Deterministic coordinate/golden search over (a, ph, w)."""
    best_h = _smooth_step(1.0, 0.0, 0.0)
    best_v = _value(best_h)

    def _cur(p):
        return _value(_smooth_step(*p))

    p = (1.0, 0.0, 0.0)
    # iterate coordinate refinements until no improvement
    for _ in range(6):
        new = list(p)
        improved = False
        # amplitude in [0.4, 1.0]
        v, x = _golden(lambda t: _cur((t, p[1], p[2])), 0.4, 1.0, 45)
        if v < best_v - 1e-12:
            new[0] = x
            improved = True
        # phase in [0, 1]
        v, x = _golden(lambda t: _cur((p[0], t, p[2])), 0.0, 1.0, 45)
        if v < best_v - 1e-12:
            new[1] = x
            improved = True
        # (w unused for the plain step; keep param for interface)
        p = tuple(new)
        h = _smooth_step(*p)
        v = _value(h)
        if v < best_v:
            best_v, best_h = v, h
        if not improved:
            break
    return best_h, best_v


def _polish(h0, v0, deadline):
    """Nelder-Mead polish over the 3 params, respecting the deadline."""
    from scipy.optimize import minimize

    best_h, best_v = h0, v0

    def obj(p):
        return _value(_smooth_step(p[0], p[1], p[2]))

    # seed the simplex from the current best params
    a, ph, w = 1.0, 0.0, 0.0
    # recover params by matching (fallback to grid)
    for aa in np.linspace(0.7, 1.0, 20):
        for pp in np.linspace(0.0, 1.0, 20):
            if _value(_smooth_step(aa, pp, 0.0)) < v0 + 1e-9:
                a, ph, w = aa, pp, 0.0
                break
        if a == 1.0 and v0 < 1.0:
            pass
    x0 = [a, ph, w]
    x0[0] = min(max(x0[0], 0.6), 1.0)
    x0[1] = min(max(x0[1], 0.0), 1.0)

    res = minimize(
        obj,
        x0,
        method="Nelder-Mead",
        bounds=[(0.6, 1.0), (0.0, 1.0), (0.0, 0.0)],
        options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9,
                 "adaptive": True},
    )
    if res.x is not None:
        h = _smooth_step(float(res.x[0]), float(res.x[1]), 0.0)
        if _valid(h):
            v = _value(h)
            if v < best_v:
                best_v, best_h = v, h
    return best_h, best_v


def _random_restart(deadline, best_v):
    """Try a few randomized smooth-step shapes; return best found."""
    import random
    best_h, best_v = None, best_v
    rng = random.Random(12345)
    for _ in range(2000):
        if time.time() >= deadline:
            break
        period = rng.choice([N, N, N, 2 * N, 3 * N])
        ph = rng.uniform(0.0, period)
        a = rng.uniform(0.7, 1.0)
        h = _smooth_step2(a, ph, 0.0, period)
        if not _valid(h):
            continue
        v = _value(h)
        if v < best_v:
            best_v, best_h = v, h
    return best_h, best_v


def construct():
    # 1) deterministic core (always valid, >= 1.0)
    h, v = _optimize_smooth_step()
    if not _valid(h):
        h = np.full(N, 0.5)
        v = _value(h)

    # 2) polish with Nelder-Mead if time remains
    if time.time() < _DEADLINE - 20.0:
        h, v = _polish(h, v, _DEADLINE)

    # 3) randomized restarts to escape any local minimum
    if time.time() < _DEADLINE - 5.0:
        h2, v2 = _random_restart(_DEADLINE, v)
        if h2 is not None and v2 < v:
            h, v = h2, v2

    # final safety: guarantee validity
    if not _valid(h):
        h = np.full(N, 0.5)
    return h
