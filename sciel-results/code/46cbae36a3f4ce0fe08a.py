"""Parametric sinusoid model projected onto feasible hyperplane, optimized with SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _raw_value(h):
    """The evaluator's objective: max cross-correlation / N * 2."""
    return float(np.max(np.correlate(h, 1.0 - h, mode="full")) / N * 2.0)


def _soft_max_value(h, beta=50.0):
    """Smooth surrogate for the max-correlation objective using log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    # Soft max via log-sum-exp
    m = np.max(corr)
    if m > 0:
        log_sum = m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr))
    else:
        log_sum = np.max(corr)  # fallback if all non-positive
    return float(log_sum / N * 2.0)


def _project(h):
    """Project h onto [0,1]^N with sum(h)=N/2 (to 1e-9*N)."""
    h = np.clip(h, 0.0, 1.0)
    # Fix sum to exactly N/2 via small uniform adjustment, then re-clip.
    # Use iterative water-filling style correction for exactness.
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        d = target - s
        if abs(d) < 1e-15:
            break
        # distribute d where we can (not at bounds)
        can_inc = h < 1.0 - 1e-15
        can_dec = h > 1e-15
        if d > 0:
            idx = np.where(can_inc)[0]
            if len(idx) == 0:
                break
            step = min(d / len(idx), (1.0 - h[idx]).min())
            h[idx] += step
        else:
            idx = np.where(can_dec)[0]
            if len(idx) == 0:
                break
            step = min(-d / len(idx), h[idx].min())
            h[idx] -= step
    # final exact fix: nudge interior points
    s = h.sum()
    d = target - s
    if abs(d) > 1e-16:
        interior = (h > 1e-12) & (h < 1.0 - 1e-12)
        idx = np.where(interior)[0]
        if len(idx) > 0:
            per = d / len(idx)
            # ensure stays in bounds
            lo = max(per, -h[idx].min() + 1e-15)
            hi = min(per, (1.0 - h[idx]).min() - 1e-15)
            per = np.clip(per, lo, hi)
            h[idx] += per
    return h


def _make_h(params):
    """params: [a1,p1, a2,p2, a3,p3, offset] -> h before projection."""
    a1, p1, a2, p2, a3, p3, off = params
    x = np.arange(N)
    h = off + a1 * np.sin(2 * np.pi * p1 * x / N) + \
            a2 * np.sin(2 * np.pi * p2 * x / N) + \
            a3 * np.sin(2 * np.pi * p3 * x / N)
    return h


def _objective(params):
    h = _make_h(params)
    h = _project(h)
    return _soft_max_value(h, beta=80.0)


def _refine_full(h0):
    """Polish a full 100-dim h via gradient-free local search on the max-correlation."""
    best = h0.copy()
    bestv = _raw_value(best)
    rng = np.random.default_rng(12345)
    budget = 10.0  # seconds
    t_start = time.time()
    step = 0.03
    n = N
    for it in range(30000):
        if time.time() - t_start > budget:
            break
        # random coordinate perturbation
        i = rng.integers(0, n)
        delta = rng.uniform(-step, step)
        cand = best.copy()
        cand[i] = np.clip(best[i] + delta, 0.0, 1.0)
        # fix sum
        s = cand.sum()
        d = n / 2.0 - s
        if abs(d) > 1e-12:
            interior = (cand > 1e-9) & (cand < 1.0 - 1e-9)
            idx = np.where(interior)[0]
            if len(idx) > 0:
                per = d / len(idx)
                lo = max(per, -cand[idx].min() + 1e-12)
                hi = min(per, (1.0 - cand[idx]).min() - 1e-12)
                per = np.clip(per, lo, hi)
                cand[idx] += per
        v = _raw_value(cand)
        if v < bestv - 1e-12:
            best = cand
            bestv = v
            step = min(step * 1.05, 0.15)
        else:
            step = max(step * 0.97, 1e-5)
    return best, bestv


def _slsqp_from(params0):
    """Run SLSQP on the 7-dim param space; return best h."""
    # bounds on params
    bounds = [(-0.6, 0.6), (0.0, 5.0),
              (-0.6, 0.6), (0.0, 5.0),
              (-0.6, 0.6), (0.0, 5.0),
              (0.0, 1.0)]
    res = minimize(_objective, params0, method="SLSQP", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-12})
    h = _project(_make_h(res.x))
    return h, _raw_value(h)


def construct():
    global _t0
    if _elapsed() > DEADLINE:
        return np.full(N, 0.5)

    rng = np.random.default_rng(100)
    best_h = np.full(N, 0.5)
    best_v = _raw_value(best_h)

    # Multi-start in param space
    n_starts = 50
    for s in range(n_starts):
        if _elapsed() > DEADLINE - 12:
            break
        # random init: small amplitudes, random phases, random freqs
        a = rng.uniform(-0.5, 0.5, size=3)
        p = rng.uniform(0.5, 4.5, size=3)
        off = rng.uniform(0.3, 0.7)
        params0 = np.array([a[0], p[0], a[1], p[1], a[2], p[2], off])
        try:
            h, v = _slsqp_from(params0)
        except Exception:
            continue
        # Use raw value for comparison
        v_raw = _raw_value(h)
        if np.isfinite(v_raw) and v_raw < best_v:
            best_v = v_raw
            best_h = h.copy()

    # Polish the best with full-dim local search
    if _elapsed() < DEADLINE - 4:
        h2, v2 = _refine_full(best_h)
        if np.isfinite(v2) and v2 < best_v:
            best_v = v2
            best_h = h2
        # Second polish pass with tighter step
        if _elapsed() < DEADLINE - 2:
            h3, v3 = _refine_full(best_h)
            if np.isfinite(v3) and v3 < best_v:
                best_v = v3
                best_h = h3

    # Ensure exactness
    best_h = _project(best_h)
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
    return best_h
