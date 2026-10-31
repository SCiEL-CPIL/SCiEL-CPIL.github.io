"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Structural change: abandon the block-enumeration family. Instead optimize the
ACTUAL discrete overlap directly. Two complementary strategies:
  1) A smooth (softmax) objective over the full h vector, solved with L-BFGS-B
     from a sinusoidal global start (avoids local minima of the piecewise max).
  2) A structured sinusoidal ansatz h_i = 0.5 + A*sin(2*pi*i/N) + B*sin(2*pi*(i+1)/N)
     whose overlap is a 1D trigonometric polynomial; minimize its exact max.
"""
import time
import math
import numpy as np

N = 100
DEADLINE = time.time() + 130.0   # safety margin before the 150s kill


def _score(h):
    """Raw max overlap of h with shifted (1-h); h zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _smooth_max(overlap, tau):
    """Softmax (smooth max) of a non-negative array."""
    m = np.max(overlap)
    if m <= 0.0:
        return 0.0
    z = overlap / tau
    return m + tau * np.log(np.sum(np.exp(z - np.max(z))))


def _smooth_objective(x, tau):
    """Smooth overlap objective for a raw (unclipped) parameter vector x."""
    h = np.clip(x, 0.0, 1.0)
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return _smooth_max(overlap, tau)


def _opt_full(tau, x0, budget):
    """L-BFGS-B on the smooth objective, with a sum-penalty barrier."""
    target = N / 2.0
    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)

    def f(x):
        h = np.clip(x, 0.0, 1.0)
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        s = _smooth_max(overlap, tau)
        # quadratic penalty for sum deviation (keeps h summing to N/2)
        s += 1000.0 * (h.sum() - target) ** 2
        return s

    from scipy.optimize import minimize
    res = minimize(f, x0, method='L-BFGS-B',
                   bounds=[(0.0, 1.0)] * N,
                   options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10,
                            'maxls': 50})
    h = np.clip(res.x, 0.0, 1.0)
    # repair exact sum
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        idx = np.argmax(h < 1.0) if (h < 1.0).any() else 0
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        # redistribute any residual over the slack
        diff = target - h.sum()
        if abs(diff) > 1e-15:
            slack = np.where((h > 1e-6) & (h < 1.0 - 1e-6))[0]
            if len(slack):
                h[slack] += diff / len(slack)
                h = np.clip(h, 0.0, 1.0)
    return h, _score(h)


def _sinusoidal_overlap(A, B):
    """Exact max overlap for h_i = 0.5 + A sin(2pi i/N) + B sin(2pi (i+1)/N)."""
    i = np.arange(N, dtype=float)
    h = 0.5 + A * np.sin(2.0 * math.pi * i / N) + B * np.sin(2.0 * math.pi * (i + 1) / N)
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap)), h


def _opt_sinusoidal():
    """Minimize the exact sinusoidal max overlap over (A, B)."""
    best = (float('inf'), None)
    grid = np.linspace(-0.45, 0.45, 61)
    for A in grid:
        for B in grid:
            s, h = _sinusoidal_overlap(A, B)
            if s < best[0]:
                best = (s, (A, B, h.copy()))
    # refine around the best with scipy
    A0, B0, _ = best[1]
    from scipy.optimize import minimize

    def obj(v):
        s, _ = _sinusoidal_overlap(v[0], v[1])
        return s
    res = minimize(obj, [A0, B0], method='Nelder-Mead',
                   options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
    s, h = _sinusoidal_overlap(res.x[0], res.x[1])
    if s < best[0]:
        best = (s, (res.x[0], res.x[1], h))
    return best


def construct():
    best_h = None
    best_score = float('inf')

    def consider(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # ---- 1) structured sinusoidal ansatz (global, cheap, strong lower bound) ----
    if time.time() < DEADLINE:
        s, (A, B, h) = _opt_sinusoidal()
        consider(h)

    # ---- 2) smooth full-vector L-BFGS-B from several starts ----
    starts = []
    # flat-ish start
    starts.append(np.full(N, 0.5))
    # sinusoidal start (use the ansatz optimum if we have it)
    if best_h is not None:
        starts.append(best_h.copy())
    # random smooth starts
    rng = np.random.default_rng(1234)
    for _ in range(4):
        A = rng.uniform(-0.4, 0.4)
        B = rng.uniform(-0.4, 0.4)
        i = np.arange(N, dtype=float)
        s0 = 0.5 + A * np.sin(2.0 * math.pi * i / N) + B * np.sin(2.0 * math.pi * (i + 1) / N)
        starts.append(np.clip(s0, 0.0, 1.0))

    taus = [0.03, 0.015, 0.008]
    for tau in taus:
        if time.time() > DEADLINE - 15.0:
            break
        for x0 in starts:
            if time.time() > DEADLINE - 15.0:
                break
            try:
                h, s = _opt_full(tau, x0, budget=None)
                consider(h)
            except Exception:
                pass

    # ---- 3) fallback ----
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N // 2] = A
        h[N // 2:] = B
        best_h = h

    # final exact-sum guarantee
    target = N / 2.0
    diff = target - best_h.sum()
    if abs(diff) > 1e-15:
        slack = np.where((best_h > 1e-9) & (best_h < 1.0 - 1e-9))[0]
        if len(slack):
            best_h[slack] += diff / len(slack)
            best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
