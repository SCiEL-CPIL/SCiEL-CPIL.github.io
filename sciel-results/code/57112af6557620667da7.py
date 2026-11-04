"""Minimize normalized max autoconvolution of a nonnegative step function on 600 intervals.

Representation: nonnegative superposition of disjoint, variable-width rectangular
pulses (positions, widths, heights), optimized with differential evolution, then
polished at the bin level with L-BFGS-B on a soft-max surrogate.

Normalization (the structural key): ratio = max(autoconv) / (integral f)^2.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600
W = 1.0 / N                       # interval width (domain [-1/4,1/4])
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# ---- fast fine-grid convolution (precomputed FFT) ----
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()


def _fine_conv(h):
    """Autoconvolution of the fine-grid lift of h (already DX-scaled), length 2*FINE_N-1."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, n=N_FFT)[:2 * FINE_N - 1]
    return conv * DX


def ratio_from_h(h):
    """max(autoconv) / (integral f)^2  -- the normalized maximum."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum() * W
    if s < 1e-15:
        return float('inf')
    conv = _fine_conv(h)
    return float(np.max(conv) / (s * s))


def soft_ratio(h, beta=80.0):
    """Smooth surrogate of ratio_from_h for differentiable optimization."""
    h = np.clip(h, 0.0, None)
    s = h.sum() * W
    if s < 1e-15:
        return 1e10
    conv = _fine_conv(h)
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    soft = m + np.log(np.sum(np.exp(conv - m))) / beta
    return float(soft / (s * s))


def _pulse_h(params, K):
    """Nonnegative heights from K rectangular pulses.
    params layout per pulse: [height, center, halfwidth]."""
    h = np.zeros(N)
    for i in range(K):
        hi = params[3 * i]
        c = np.clip(params[3 * i + 1], -0.25, 0.25)
        hw = np.clip(params[3 * i + 2], 0.0, 0.25)
        m = (BIN_CENTERS >= c - hw) & (BIN_CENTERS <= c + hw)
        h[m] += hi
    return h


def _pulse_obj(params, K):
    v = ratio_from_h(_pulse_h(params, K))
    return v if np.isfinite(v) else 1e10


def _phase_pulse(K, deadline):
    """Differential evolution over K rectangular pulses."""
    lb = np.array([0.0, -0.25, 0.0], dtype=float)
    ub = np.array([3.0, 0.25, 0.25], dtype=float)
    bounds = [(lb[j], ub[j]) for j in range(3) for _ in range(K)]
    best = [1e10, None]

    def update(res):
        v = _pulse_obj(res, K)
        if v < best[0]:
            best[0] = v
            best[1] = _pulse_h(res, K).copy()

    def wrap(res):
        update(res)
        return _pulse_obj(res, K)

    try:
        differential_evolution(wrap, bounds, maxiter=300, popsize=18,
                               tol=1e-6, seed=42, polish=False,
                               updating='deferred', mutation=(0.4, 1.0),
                               recombination=0.8, callback=lambda res, k:
                               (time.time() > deadline - 8.0) and res or None)
    except Exception:
        pass
    # also evaluate the final best
    return best[0], best[1]


def _phase_lbfgs(h0, deadline):
    """L-BFGS-B on bin-level soft-max surrogate (cheap FFT gradient)."""
    bounds = [(0.0, None)] * N
    try:
        res = minimize(soft_ratio, h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() > deadline - 4.0:
            return h0
        h = np.clip(res.x, 0.0, None)
        v = ratio_from_h(h)
        return (v, h) if np.isfinite(v) else (float('inf'), h0)
    except Exception:
        return (float('inf'), h0)


def construct():
    start = time.time()
    deadline = start + 120.0        # safety margin before 150s kill

    best_h = np.ones(N, dtype=np.float64)
    best_v = ratio_from_h(best_h)

    def update(h, v):
        nonlocal best_h, best_v
        if v < best_v and np.isfinite(v):
            best_v = v
            best_h = h.copy()

    # Phase 1: rectangular pulses via differential evolution (new representation)
    for K in (4, 6, 8):
        if time.time() > deadline - 45.0:
            break
        v, h = _phase_pulse(K, deadline)
        if h is not None:
            update(h, v)

    # Phase 2: bin-level L-BFGS-B polish on the best pulse result
    if time.time() < deadline - 15.0:
        v, h = _phase_lbfgs(best_h, deadline)
        update(h, v)

    # Phase 3: small random local search polish
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(777)
        for _ in range(2500):
            if time.time() > deadline - 2.0:
                break
            n_p = int(rng.integers(3, 12))
            idx = rng.choice(N, size=n_p, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng.normal(0, 0.02, size=n_p)
            h_try = np.clip(h_try, 0, None)
            update(h_try, ratio_from_h(h_try))

    # safety
    best_h = np.asarray(best_h, dtype=float)
    if best_h.shape != (N,):
        return [1.0] * N
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=0.0, neginf=0.0)
    best_h = np.maximum(best_h, 0.0)
    if not np.isfinite(ratio_from_h(best_h)):
        return [1.0] * N
    return [float(x) for x in best_h]
