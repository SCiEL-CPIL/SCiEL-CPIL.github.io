"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound B(a) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over the three free Hermite coefficients (degrees 0, 4, 8). The degree-12
weight is fixed so that g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 25.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    h0 = np.ones_like(x)
    h4 = 16.0 * x2**2 - 48.0 * x2 + 12.0
    h8 = 256.0 * x2**4 - 3584.0 * x2**3 + 13440.0 * x2**2 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x2**6 - 212992.0 * x2**5 + 3317760.0 * x2**4
           - 25333440.0 * x2**3 + 100810080.0 * x2**2 - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_fast(a, n_coarse=5000):
    """Fast approximate bound using a moderate-resolution grid."""
    grid = np.concatenate([
        np.linspace(0.0, 10.0, n_coarse),
        np.geomspace(10.0, 40.0, n_coarse // 4)
    ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.where(ratio > 0, ratio, 1e-18)
    return float(np.max(ratio))


def _bound_smooth(a, temperature=0.05):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    grid = np.concatenate([
        np.linspace(0.0, 8.0, 15000),
        np.geomspace(8.0, 35.0, 7500)
    ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-10, 1e15)
    
    m = float(np.max(ratio))
    if m > 1e14:
        return m
    
    shifted = ratio - m
    log_sum_exp = m + temperature * math.log(float(np.mean(np.exp(shifted / temperature))))
    return log_sum_exp


def _bound_exact(a):
    """Exact max of (1-g)/g over lambda>=0 with adaptive local refinement."""
    grid = np.concatenate([
        np.linspace(0.0, 20.0, 50001),
        np.geomspace(1e-10, 40.0, 2500),
    ])
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.where(ratio > 0, ratio, 1e-18)
    
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    
    # Adaptive local refinement
    for level in range(5):
        if remaining() < 1.0:
            break
        width = 0.05 * (0.5 ** level)
        lo = max(0.0, x0 - width)
        hi = x0 + width
        local = np.linspace(lo, hi, 3001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-15, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.where(rl > 0, rl, 1e-18)
        
        local_max = float(np.max(rl))
        local_idx = int(np.argmax(rl))
        x0 = float(local[local_idx])
        
        if local_max > val:
            val = local_max
    
    return val


def remaining():
    if _START is None:
        return _DEADLINE - _MARGIN
    return _DEADLINE - _MARGIN - (time.time() - _START)


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    from scipy.optimize import minimize, differential_evolution

    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    # Phase 1: Global exploration with differential evolution on fast objective
    if remaining() > 100.0:
        try:
            bounds = [(-2.5, 0.0), (-0.15, 0.15), (-0.01, 0.01)]
            de_result = differential_evolution(
                _bound_fast, bounds,
                seed=42, maxiter=50, popsize=20, tol=1e-8,
                polish=False, init='sobol',
                updating='deferred',
            )
            if de_result.fun < best_val:
                cand = np.asarray(de_result.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS-B on smooth surrogate with decreasing temperature
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [best_a.copy()]
    
    # Structured grid around seed
    for da0 in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
        for da4 in (-0.02, -0.01, 0.0, 0.01, 0.02):
            for da8 in (-0.0005, 0.0, 0.0005):
                s = best_a + np.array([da0, da4, da8])
                if s[0] < 0:  # a0 must be negative
                    starts.append(s.copy())
    
    # Random perturbations
    for scale in (0.01, 0.03, 0.05, 0.1):
        starts.append(best_a + scale * rng.standard_normal(3))
    
    # Temperature schedule: coarse to fine
    temps = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temps:
        for s in starts:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-12, "maxls": 40},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)) and cand[0] < 0:
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue

    # Phase 3: Final polish with Nelder-Mead on exact objective
    if remaining() > 8.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-12, "fatol": 1e-12, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)) and cand[0] < 0:
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 4: Coordinate descent polish
    if remaining() > 4.0:
        for step in (0.001, 0.0005, 0.0002, 0.0001):
            if remaining() < 2.0:
                break
            improved = True
            while improved and remaining() > 1.0:
                improved = False
                for axis in range(3):
                    for direction in (-1, 1):
                        cand = best_a.copy()
                        cand[axis] += direction * step
                        if cand[0] >= 0:
                            continue
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
                            improved = True

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
