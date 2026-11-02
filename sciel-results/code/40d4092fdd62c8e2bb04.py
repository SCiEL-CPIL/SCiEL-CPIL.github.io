"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) over the three free
Hermite coefficients (degrees 0, 4, 8).  The degree-12 weight is fixed so that
g(0) = 0.  The bound is sqrt(2*pi) * (sup |x*g(x)|)^2, computed on a dense
grid with local refinement near the argmax.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0, H_4, H_8, H_12 at points x."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    h0 = np.ones_like(x)
    h4 = 16.0 * x4 - 48.0 * x2 + 12.0
    h8 = 256.0 * x8 - 3584.0 * x6 + 13440.0 * x4 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x12 - 212992.0 * x10 + 3317760.0 * x8
           - 25333440.0 * x6 + 100810080.0 * x4 - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _compute_bound(a, grid=None):
    """Compute the upper bound sqrt(2*pi) * (sup |x*g(x)|)^2."""
    if grid is None:
        grid = np.linspace(-12.0, 12.0, 24001)
    g = _g_of_lambda(a, grid)
    xg = grid * g
    abs_xg = np.abs(xg)
    # Local refinement near argmax
    idx = int(np.argmax(abs_xg))
    x0 = float(grid[idx])
    val = float(abs_xg[idx])
    # Refine locally
    local = np.linspace(max(-12.0, x0 - 0.05), min(12.0, x0 + 0.05), 4001)
    gl = _g_of_lambda(a, local)
    xgl = local * gl
    abs_xgl = np.abs(xgl)
    val = max(val, float(np.max(abs_xgl)))
    # Also check the negative side (since x*g can have peaks on both sides)
    idx_neg = int(np.argmin(abs_xg))  # This won't work for negative, let's do it properly
    # Actually, let's just take the max over the whole refined grid
    # Re-do with a broader local search
    local2 = np.linspace(max(-12.0, x0 - 0.2), min(12.0, x0 + 0.2), 8001)
    gl2 = _g_of_lambda(a, local2)
    xgl2 = local2 * gl2
    val = max(val, float(np.max(np.abs(xgl2))))
    return math.sqrt(2.0 * math.pi) * val * val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth surrogate using log-sum-exp for the max of |x*g(x)|."""
    if grid is None:
        grid = np.linspace(-12.0, 12.0, 12001)
    g = _g_of_lambda(a, grid)
    xg = np.abs(grid * g)
    # Clip for numerical stability
    xg = np.clip(xg, 1e-15, 1e15)
    m = float(np.max(xg))
    s = np.exp((xg - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return math.sqrt(2.0 * math.pi) * softmax * softmax


def construct():
    global _START
    if _START is None:
        _START = time.time()
    best_a = _SEED.copy()
    best_val = _compute_bound(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize

    # Phase 1: Multi-start global optimization on the smooth surrogate
    rng = np.random.default_rng(0)
    starts = [_SEED.copy()]
    for scale in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
        starts.append(_SEED + scale * 0.015 * rng.standard_normal(3))
    for _ in range(6):
        starts.append(rng.standard_normal(3) * 0.03)
    starts.append(np.array([-1.1, 0.04, 0.0002]))
    starts.append(np.array([-1.05, 0.03, 0.0004]))
    starts.append(np.array([-1.15, 0.05, 0.0001]))
    starts.append(np.array([-1.0, 0.02, 0.0006]))
    starts.append(np.array([-1.2, 0.03, 0.00025]))
    starts.append(np.array([-1.0, 0.05, 0.0004]))

    for temp in (0.03, 0.01):
        for s in starts:
            if remaining() < 4.0:
                break
            try:
                res = minimize(
                    lambda x: _bound_smooth(x, temperature=temp), s, method="L-BFGS-B",
                    options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-8},
                )
            except Exception:
                continue
            cand = np.asarray(res.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            v = _compute_bound(cand)
            if v < best_val:
                best_val = v
                best_a = cand.copy()

    # Phase 2: High-precision polish on the exact objective
    if remaining() > 5.0:
        try:
            res = minimize(
                _compute_bound, best_a, method="L-BFGS-B",
                options={"maxiter": 600, "ftol": 1e-14, "gtol": 1e-10},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _compute_bound(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead refinement
    if remaining() > 5.0:
        try:
            res = minimize(
                _compute_bound, best_a, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-12, "fatol": 1e-14, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _compute_bound(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 4: Adaptive local grid polish
    if remaining() > 2.0:
        for step in (0.005, 0.002, 0.001):
            if remaining() < 1.0:
                break
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 0.5:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _compute_bound(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
