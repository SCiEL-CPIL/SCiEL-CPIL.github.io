"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the classical upper bound on A(f)*A(Fourier(f)) over the three free
Hermite coefficients (degrees 0, 4, 8).  The degree-12 weight is fixed so that
g(0) = 0.  We minimize the bound
    B(a) = max_{lambda>=0} (1 - g(lambda)) / g(lambda)
using a smooth log-sum-exp surrogate for the outer max during optimization,
then a high-precision polish on the exact (max) objective.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 25.0          # safety margin (seconds) before the hard kill
_START = None


def _h2n(n):
    """H_0(0), H_4(0), H_8(0), H_12(0) for the physicists' Hermite polynomials."""
    return 1.0, 12.0, 1680.0, 798336.0


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    h8 = (256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4
          - 13440.0 * x**2 + 1680.0)
    h12 = (4096.0 * x**12 - 212992.0 * x**10 + 3317760.0 * x**8
           - 25333440.0 * x**6 + 100810080.0 * x**4
           - 201620160.0 * x**2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid, plus a coarse
    refinement near the argmax."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 30.0, 200001),
            np.geomspace(1e-6, 60.0, 4000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    if 0.0 < x0 < 60.0:
        local = np.linspace(max(0.0, x0 - 0.05), x0 + 0.05, 4001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-12, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        val = max(val, float(np.max(rl)))
    return val


def _bound_smooth(a, grid=None, temperature=0.02):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.0, 30.0, 60001)
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


def _bound_smooth_analytic(a, temperature=0.02):
    """Smooth surrogate using a smaller grid for speed during optimization."""
    grid = np.linspace(0.0, 20.0, 20001)
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
    if _START is None:
        _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize, differential_evolution

    # --- Phase 1: Global search with differential evolution on smooth surrogate ---
    if remaining() > 30.0:
        try:
            bounds = [(-2.0, 0.0), (-0.1, 0.1), (-0.001, 0.001)]
            result_de = differential_evolution(
                lambda x: _bound_smooth_analytic(x, temperature=0.05),
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                polish=True,
            )
            cand = np.asarray(result_de.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # --- Phase 2: Multi-start local optimization on smooth surrogate ---
    rng = np.random.default_rng(0)
    starts = [best_a.copy()]
    # Add structured starts around the seed and current best
    for scale in (0.01, 0.03, 0.05, 0.1):
        starts.append(best_a + scale * rng.standard_normal(3))
        starts.append(_SEED + scale * rng.standard_normal(3))
    # Add some hand-picked starts
    starts.append(np.array([-1.1, 0.04, 0.0002]))
    starts.append(np.array([-1.05, 0.035, 0.0003]))
    starts.append(np.array([-1.15, 0.045, 0.00015]))
    starts.append(np.array([-1.08, 0.042, 0.00025]))
    starts.append(np.array([-1.12, 0.038, 0.00018]))

    for temp in (0.05, 0.02, 0.01):
        for s in starts:
            if remaining() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth_analytic(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
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

    # --- Phase 3: High-precision polish on the exact (max) objective ---
    if remaining() > 10.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # --- Phase 4: Adaptive local grid polish around the best point ---
    if remaining() > 3.0:
        for step in (0.003, 0.001, 0.0005):
            if remaining() < 1.0:
                break
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 0.5:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
