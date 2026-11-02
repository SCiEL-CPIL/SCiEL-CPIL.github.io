"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over three free Hermite coefficients (degrees 0, 4, 8). The degree-12 weight is
fixed so that g(0) = 0. Uses a smooth log-sum-exp surrogate for the outer max
during optimization, then a high-precision polish on the exact objective.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START = time.time()
_DEADLINE = 210.0
_MARGIN = 30.0  # safety margin before hard kill


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _hermite_phys(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence. Vectorized."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0) = 0."""
    a0, a4, a8 = a
    # H_n(0) for even n: H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=798336
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    lam = np.asarray(lam, dtype=float)
    h0 = np.ones_like(lam)
    h4 = _hermite_phys(4, lam)
    h8 = _hermite_phys(8, lam)
    h12 = _hermite_phys(12, lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with local refinement."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 25.0, 150001),
            np.geomspace(1e-6, 50.0, 3000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.maximum(ratio, 0.0)
    
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    
    # Local refinement around argmax
    if 0.0 < x0 < 50.0:
        local = np.linspace(max(0.0, x0 - 0.03), x0 + 0.03, 5001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-12, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.maximum(rl, 0.0)
        val = max(val, float(np.max(rl)))
    return val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 50001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def _bound_smooth_tight(a, grid=None, temperature=0.005):
    """Tighter smooth surrogate for final polish phase."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 80001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def construct():
    global _START
    _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    # Phase 1: Multi-start global optimization on smooth surrogate
    rng = np.random.default_rng(42)
    starts = [_SEED.copy()]
    for scale in (0.3, 0.5, 1.0, 2.0, 4.0):
        starts.append(_SEED + scale * 0.01 * rng.standard_normal(3))
    for _ in range(12):
        starts.append(rng.standard_normal(3) * 0.08)
    # Structured starts
    starts.append(np.array([-0.3, 0.01, 0.0002]))
    starts.append(np.array([-0.8, 0.03, 0.0005]))
    starts.append(np.array([-1.5, 0.06, 0.001]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-0.5, 0.02, 0.0004]))
    
    for s in starts:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                _bound_smooth, s, method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
            )
        except Exception:
            continue
        cand = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(cand)):
            continue
        v = _bound_exact(cand)
        if v < best_val:
            best_val = v
            best_a = cand.copy()
    
    # Phase 2: Polish with tighter smooth surrogate
    if _remaining() > 5.0:
        try:
            res = minimize(
                _bound_smooth_tight, best_a, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Direct optimization on exact objective (nonsmooth but 3D)
    if _remaining() > 4.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-12},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Fine grid polish around best point
    if _remaining() > 3.0:
        step = 0.001
        for i in (-2, -1, 0, 1, 2):
            for j in (-2, -1, 0, 1, 2):
                for k in (-2, -1, 0, 1, 2):
                    if _remaining() < 1.0:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    # Phase 5: Even finer local search
    if _remaining() > 2.0:
        step = 0.0003
        for i in (-3, -2, -1, 0, 1, 2, 3):
            for j in (-3, -2, -1, 0, 1, 2, 3):
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    if _remaining() < 0.5:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
