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
_MARGIN = 30.0
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


def _a12_from(a):
    a0, a4, a8 = a
    return -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)."""
    a0, a4, a8 = a
    a12 = _a12_from(a)
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _ratio(a, lam):
    g = _g_of_lambda(a, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1.0 - g) / g
    r = np.where(g > 1e-14, r, 1e18)
    r = np.where(np.isfinite(r), r, 1e18)
    return r


def _max_ratio_1d(a, lam_lo=0.0, lam_hi=8.0, n_coarse=2000):
    """Find max of ratio over [lam_lo, lam_hi] with local refinement."""
    grid = np.linspace(lam_lo, lam_hi, n_coarse)
    r = _ratio(a, grid)
    idx = int(np.argmax(r))
    val = float(r[idx])
    x0 = float(grid[idx])
    # Local refinement around the peak
    for width in (0.05, 0.01, 0.002, 0.0005):
        lo = max(lam_lo, x0 - width)
        hi = min(lam_hi, x0 + width)
        if hi - lo < 1e-9:
            break
        local = np.linspace(lo, hi, 2001)
        rl = _ratio(a, local)
        il = int(np.argmax(rl))
        new_val = float(rl[il])
        if new_val > val:
            val = new_val
            x0 = float(local[il])
    return val, x0


def _bound_exact(a):
    """Exact max of (1-g)/g over lambda>=0."""
    # Search a broad range first
    v1, x1 = _max_ratio_1d(a, 0.0, 8.0, 4000)
    v2, x2 = _max_ratio_1d(a, 0.0, 2.0, 4000)
    v3, x3 = _max_ratio_1d(a, 2.0, 10.0, 2000)
    v4, x4 = _max_ratio_1d(a, 0.0, 0.5, 2000)
    # Pick the worst
    candidates = [(v1, x1), (v2, x2), (v3, x3), (v4, x4)]
    best_v, best_x = max(candidates, key=lambda t: t[0])
    # Final fine polish around the worst point
    if 0.0 < best_x < 10.0:
        for width in (0.005, 0.001, 0.0002):
            lo = max(0.0, best_x - width)
            hi = best_x + width
            local = np.linspace(lo, hi, 2001)
            rl = _ratio(a, local)
            il = int(np.argmax(rl))
            new_v = float(rl[il])
            if new_v > best_v:
                best_v = new_v
                best_x = float(local[il])
    return best_v


def _bound_smooth(a, temperature=0.01, n_grid=8001):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    grid = np.linspace(0.0, 6.0, n_grid)
    g = _g_of_lambda(a, grid)
    min_g = float(np.min(g))
    if min_g < 1e-5:
        return 1e10 + (1e-5 - min_g) * 1e8
    r = _ratio(a, grid)
    r = np.clip(r, 1e-12, 1e15)
    m = float(np.max(r))
    s = np.exp((r - m) / temperature)
    return m + temperature * math.log(float(np.mean(s)))


def _bound_nonsmooth(a):
    """Direct max using a moderate grid (faster than _bound_exact)."""
    grid = np.linspace(0.0, 6.0, 12001)
    r = _ratio(a, grid)
    return float(np.max(r))


def construct():
    global _START
    if _START is None:
        _START = time.time()

    from scipy.optimize import minimize, differential_evolution, brent

    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    # Phase 1: Differential evolution for global exploration
    if remaining() > 60.0:
        try:
            bounds = [(-5.0, 0.0), (-0.5, 0.5), (-0.05, 0.05)]
            de_result = differential_evolution(
                _bound_smooth, bounds, args=(0.02,),
                seed=42, maxiter=80, popsize=25, tol=1e-8,
                polish=True, init='sobol',
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
    starts = [best_a.copy()]
    for scale in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        starts.append(best_a + scale * rng.standard_normal(3))

    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for temp in temps:
        for s in starts:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-11},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue

    # Phase 3: Direct nonsmooth optimization with multiple starts
    if remaining() > 10.0:
        for s in [best_a.copy(), best_a + 0.01 * rng.standard_normal(3),
                  best_a + 0.005 * rng.standard_normal(3)]:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    _bound_nonsmooth, s,
                    method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-12,
                             "adaptive": True},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue

    # Phase 4: High-precision polish with very tight smooth surrogate
    if remaining() > 5.0:
        try:
            res = minimize(
                lambda x: _bound_smooth(x, temperature=0.001, n_grid=12001), best_a,
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-13},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 5: Coordinate descent with 1D line search on each axis
    if remaining() > 3.0:
        for _ in range(3):
            if remaining() < 2.0:
                break
            improved = False
            for dim in range(3):
                if remaining() < 1.0:
                    break
                try:
                    def obj_1d(d, x):
                        xx = best_a.copy()
                        xx[dim] = x
                        return _bound_nonsmooth(xx)
                    res = brent(obj_1d, a=best_a[dim] - 0.05, b=best_a[dim] + 0.05,
                                xtol=1e-10, maxiter=100)
                    v = _bound_exact(best_a)
                    best_a[dim] = res.x
                    v = _bound_exact(best_a)
                    if v < best_val:
                        best_val = v
                        improved = True
                except Exception:
                    continue
            if not improved:
                break

    # Phase 6: Fine grid polish
    if remaining() > 2.0:
        for step in (0.001, 0.0005, 0.0002):
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
