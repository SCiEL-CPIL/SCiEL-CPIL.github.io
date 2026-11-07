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


def _project(h):
    """Project h onto [0,1]^N with sum(h)=N/2 (to 1e-9*N)."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        d = target - s
        if abs(d) < 1e-15:
            break
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
            lo = max(per, -h[idx].min() + 1e-15)
            hi = min(per, (1.0 - h[idx]).min() - 1e-15)
            per = np.clip(per, lo, hi)
            h[idx] += per
    return h


def _make_h(params):
    """params: [a1,p1, a2,p2, a3,p3, a4,p4, offset] -> h before projection."""
    a1, p1, a2, p2, a3, p3, a4, p4, off = params
    x = np.arange(N)
    h = off + a1 * np.sin(2 * np.pi * p1 * x / N) + \
            a2 * np.sin(2 * np.pi * p2 * x / N) + \
            a3 * np.sin(2 * np.pi * p3 * x / N) + \
            a4 * np.sin(2 * np.pi * p4 * x / N)
    return h


def _objective(params):
    """Use raw value directly — SLSQP handles the piecewise-smooth max well enough."""
    h = _make_h(params)
    h = _project(h)
    return _raw_value(h)


def _slsqp_from(params0):
    """Run SLSQP on the 9-dim param space; return best h."""
    bounds = [(-0.5, 0.5), (0.1, 5.0),
              (-0.5, 0.5), (0.1, 5.0),
              (-0.5, 0.5), (0.1, 5.0),
              (-0.5, 0.5), (0.1, 5.0),
              (0.0, 1.0)]
    res = minimize(_objective, params0, method="SLSQP", bounds=bounds,
                   options={"maxiter": 150, "ftol": 1e-10, "eps": 1e-8})
    h = _project(_make_h(res.x))
    return h, _raw_value(h)


def _spectral_smooth(h, alpha=0.3):
    """Apply light spectral smoothing to reduce high-frequency noise before refinement."""
    h_hat = np.fft.rfft(h)
    freqs = np.fft.rfftfreq(N)
    # Attenuate high frequencies (keep DC and low modes)
    mask = np.exp(-alpha * (freqs * N) ** 2)
    mask[0] = 1.0  # preserve DC (mean)
    h_smooth = np.fft.irfft(h_hat * mask, n=N)
    return _project(h_smooth)


def _refine_full(h0, budget=10.0, seed=12345):
    """Polish a full 100-dim h via adaptive coordinate + multi-coordinate search."""
    best = h0.copy()
    bestv = _raw_value(best)
    rng = np.random.default_rng(seed)
    t_start = time.time()
    step = 0.025
    n = N
    stagnation = 0
    for it in range(50000):
        if time.time() - t_start > budget:
            break
        # Occasionally try multi-coordinate moves (2-3 coords)
        if rng.random() < 0.15:
            n_move = int(rng.integers(2, 4))
            idxs = rng.choice(n, size=n_move, replace=False)
            deltas = rng.uniform(-step, step, size=n_move)
            cand = best.copy()
            for i, d in zip(idxs, deltas):
                cand[i] = np.clip(best[i] + d, 0.0, 1.0)
        else:
            i = rng.integers(0, n)
            delta = rng.uniform(-step, step)
            cand = best.copy()
            cand[i] = np.clip(best[i] + delta, 0.0, 1.0)
        # fix sum
        s = cand.sum()
        d_sum = n / 2.0 - s
        if abs(d_sum) > 1e-12:
            interior = (cand > 1e-9) & (cand < 1.0 - 1e-9)
            idx = np.where(interior)[0]
            if len(idx) > 0:
                per = d_sum / len(idx)
                lo = max(per, -cand[idx].min() + 1e-12)
                hi = min(per, (1.0 - cand[idx]).min() - 1e-12)
                per = np.clip(per, lo, hi)
                cand[idx] += per
        v = _raw_value(cand)
        if v < bestv - 1e-13:
            best = cand
            bestv = v
            stagnation = 0
            step = min(step * 1.03, 0.12)
        else:
            stagnation += 1
            if stagnation % 200 == 0:
                step = max(step * 0.9, 1e-5)
            # Occasionally restart step if stuck
            if stagnation > 500:
                step = 0.02
                stagnation = 0
    return best, bestv


def construct():
    global _t0
    if _elapsed() > DEADLINE:
        return np.full(N, 0.5)

    rng = np.random.default_rng(100)
    best_h = np.full(N, 0.5)
    best_v = _raw_value(best_h)

    # Multi-start in param space with 4 sinusoidal components
    n_starts = 80
    for s in range(n_starts):
        if _elapsed() > DEADLINE - 15:
            break
        # Bias toward low frequencies for better global structure
        a = rng.uniform(-0.4, 0.4, size=4)
        # Frequencies: mix of low and mid-range
        p = rng.uniform(0.5, 4.0, size=4)
        # Ensure at least one low-freq component
        if s % 3 == 0:
            p[0] = rng.uniform(0.5, 1.5)
        off = rng.uniform(0.35, 0.65)
        params0 = np.array([a[0], p[0], a[1], p[1], a[2], p[2], a[3], p[3], off])
        try:
            h, v = _slsqp_from(params0)
        except Exception:
            continue
        v_raw = _raw_value(h)
        if np.isfinite(v_raw) and v_raw < best_v:
            best_v = v_raw
            best_h = h.copy()

    # Spectral smoothing pre-pass to reduce high-freq artifacts
    if _elapsed() < DEADLINE - 12:
        h_smooth = _spectral_smooth(best_h, alpha=0.2)
        v_smooth = _raw_value(h_smooth)
        if np.isfinite(v_smooth) and v_smooth < best_v:
            best_v = v_smooth
            best_h = h_smooth

    # Polish the best with full-dim local search
    if _elapsed() < DEADLINE - 6:
        h2, v2 = _refine_full(best_h, budget=12.0, seed=12345)
        if np.isfinite(v2) and v2 < best_v:
            best_v = v2
            best_h = h2
        # Second polish pass with different seed
        if _elapsed() < DEADLINE - 3:
            h3, v3 = _refine_full(best_h, budget=8.0, seed=99999)
            if np.isfinite(v3) and v3 < best_v:
                best_v = v3
                best_h = h3

    # Ensure exactness
    best_h = _project(best_h)
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
    return best_h
