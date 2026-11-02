"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound B(a) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over the three free Hermite coefficients (degrees 0, 4, 8). The degree-12
weight is fixed so that g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.signal import find_peaks

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 15.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    h8 = 256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4 - 13440.0 * x**2 + 1680.0
    h12 = (4096.0 * x**12 - 212992.0 * x**10 + 3317760.0 * x**8
           - 25333440.0 * x**6 + 100810080.0 * x**4 - 201620160.0 * x**2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with local refinement."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 25.0, 200001),
            np.geomspace(1e-8, 50.0, 10000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    if 0.0 < x0 < 50.0:
        # Multi-scale local refinement
        for width in (0.05, 0.01, 0.002, 0.0005):
            local = np.linspace(max(0.0, x0 - width), x0 + width, 5001)
            gl = _g_of_lambda(a, local)
            with np.errstate(divide="ignore", invalid="ignore"):
                rl = (1.0 - gl) / gl
            rl = np.where(gl > 1e-14, rl, 1e18)
            rl = np.where(np.isfinite(rl), rl, 1e18)
            val = max(val, float(np.max(rl)))
            x0 = float(local[int(np.argmax(rl))])
    return val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 50001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    return m + temperature * math.log(float(np.mean(s)))


def _bound_smooth_grad(a, grid=None, temperature=0.01):
    """
    Smooth surrogate with analytical gradient.
    
    Uses log-sum-exp formulation: f(a) = m + T * log(mean(exp((r-m)/T)))
    where r(lambda) = (1-g)/g and g depends linearly on a through the polynomial.
    """
    if grid is None:
        grid = np.linspace(0.0, 25.0, 30001)
    
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    
    h0, h4, h8, h12 = _hermite_values(grid)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-0.5 * grid * grid)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    
    m = float(np.max(ratio))
    exp_terms = np.exp((ratio - m) / temperature)
    mean_exp = float(np.mean(exp_terms))
    val = m + temperature * math.log(mean_exp)
    
    # Weights for the log-sum-exp gradient
    w = exp_terms / mean_exp  # normalized weights
    
    # dr/da_i = -dg/da_i / g^2
    exp_term = np.exp(-0.5 * grid * grid)
    da12_da0 = -1.0 / 798336.0
    da12_da4 = -12.0 / 798336.0
    da12_da8 = -1680.0 / 798336.0
    
    dg_da0 = (h0 + da12_da0 * h12) * exp_term
    dg_da4 = (h4 + da12_da4 * h12) * exp_term
    dg_da8 = (h8 + da12_da8 * h12) * exp_term
    
    with np.errstate(divide="ignore", invalid="ignore"):
        dr_da0 = -dg_da0 / (g * g)
        dr_da4 = -dg_da4 / (g * g)
        dr_da8 = -dg_da8 / (g * g)
    
    dr_da0 = np.where(np.isfinite(dr_da0), dr_da0, 0.0)
    dr_da4 = np.where(np.isfinite(dr_da4), dr_da4, 0.0)
    dr_da8 = np.where(np.isfinite(dr_da8), dr_da8, 0.0)
    
    grad = np.array([
        temperature * float(np.sum(w * dr_da0)),
        temperature * float(np.sum(w * dr_da4)),
        temperature * float(np.sum(w * dr_da8)),
    ])
    
    return val, grad


def _epigraph_objective(params, grid=None):
    """
    Epigraph formulation: minimize t subject to t >= (1-g(lambda))/g(lambda) for all lambda.
    params = [a0, a4, a8, t]
    """
    a0, a4, a8, t = params
    a = np.array([a0, a4, a8])
    
    if grid is None:
        grid = np.linspace(0.0, 25.0, 20001)
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    
    # Penalty for constraint violation: sum of max(0, ratio - t)^2
    violation = np.maximum(0, ratio - t)
    penalty = float(np.sum(violation ** 2))
    
    return t + 1e6 * penalty


def _epigraph_grad(params, grid=None):
    """Gradient for epigraph objective."""
    a0, a4, a8, t = params
    a = np.array([a0, a4, a8])
    
    if grid is None:
        grid = np.linspace(0.0, 25.0, 20001)
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    
    violation = np.maximum(0, ratio - t)
    
    # d/dt of objective = 1 + 2e6 * sum(violation * (-1)) = 1 - 2e6 * sum(violation)
    grad_t = 1.0 - 2e6 * float(np.sum(violation))
    
    # d/da_i of objective = 2e6 * sum(violation * dr/da_i)
    h0, h4, h8, h12 = _hermite_values(grid)
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    
    exp_term = np.exp(-0.5 * grid * grid)
    da12_da0 = -1.0 / 798336.0
    da12_da4 = -12.0 / 798336.0
    da12_da8 = -1680.0 / 798336.0
    
    dg_da0 = (h0 + da12_da0 * h12) * exp_term
    dg_da4 = (h4 + da12_da4 * h12) * exp_term
    dg_da8 = (h8 + da12_da8 * h12) * exp_term
    
    with np.errstate(divide="ignore", invalid="ignore"):
        dr_da0 = -dg_da0 / (g * g)
        dr_da4 = -dg_da4 / (g * g)
        dr_da8 = -dg_da8 / (g * g)
    
    dr_da0 = np.where(np.isfinite(dr_da0), dr_da0, 0.0)
    dr_da4 = np.where(np.isfinite(dr_da4), dr_da4, 0.0)
    dr_da8 = np.where(np.isfinite(dr_da8), dr_da8, 0.0)
    
    grad_a0 = 2e6 * float(np.sum(violation * dr_da0))
    grad_a4 = 2e6 * float(np.sum(violation * dr_da4))
    grad_a8 = 2e6 * float(np.sum(violation * dr_da8))
    
    return np.array([grad_a0, grad_a4, grad_a8, grad_t])


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)
    
    # Phase 1: Differential evolution for global exploration
    if remaining() > 100.0:
        try:
            bounds = [(-2.0, 0.0), (-0.1, 0.1), (-0.01, 0.01)]
            de_result = differential_evolution(
                _bound_smooth, bounds, args=(None, 0.1),
                seed=42, maxiter=80, popsize=30, tol=1e-10,
                polish=True, init='sobol',
            )
            if de_result.success or de_result.fun < best_val:
                cand = np.asarray(de_result.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B on smooth surrogate with analytical gradient
    rng = np.random.default_rng(123)
    starts = [best_a.copy()]
    for scale in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        starts.append(best_a + scale * rng.standard_normal(3))
    # Structured starts around seed region
    for a0 in (-1.15, -1.13, -1.12, -1.11, -1.10, -1.09, -1.08, -1.05, -1.0, -0.95, -0.9):
        for a4 in (0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0, -0.005):
            for a8 in (0.0005, 0.0004, 0.0003, 0.00025, 0.0002, 0.00015, 0.0001, 0.00005, 0.0, -0.00005, -0.0001):
                starts.append(np.array([a0, a4, a8]))
    
    # Use analytical gradient for faster convergence with adaptive temperature
    temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for temp in temperatures:
        for s in starts:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda x: _bound_smooth_grad(x, temperature=temp)[0],
                    s,
                    jac=lambda x: _bound_smooth_grad(x, temperature=temp)[1],
                    method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-11, "maxls": 50},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue
    
    # Phase 3: Epigraph formulation for more robust optimization
    if remaining() > 30.0:
        try:
            # Start from current best
            t0 = best_val
            x0 = np.concatenate([best_a, [t0]])
            
            res = minimize(
                _epigraph_objective, x0,
                jac=_epigraph_grad,
                method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
            )
            cand_full = np.asarray(res.x, dtype=float)
            cand = cand_full[:3]
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 4: High-precision polish on exact objective
    if remaining() > 15.0:
        # Find the critical point more precisely
        grid_fine = np.linspace(0.0, 20.0, 100001)
        g_fine = _g_of_lambda(best_a, grid_fine)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_fine = (1.0 - g_fine) / g_fine
        ratio_fine = np.where(g_fine > 1e-14, ratio_fine, 1e18)
        ratio_fine = np.where(np.isfinite(ratio_fine), ratio_fine, 1e18)
        
        # Find all local maxima of the ratio
        peaks, _ = find_peaks(ratio_fine, height=best_val * 0.95, distance=50)
        
        # Refine around each peak
        for p in peaks[:15]:
            if remaining() < 5.0:
                break
            x_peak = float(grid_fine[p])
            # Very fine local grid
            local = np.linspace(max(0.0, x_peak - 0.005), x_peak + 0.005, 20001)
            gl = _g_of_lambda(best_a, local)
            with np.errstate(divide="ignore", invalid="ignore"):
                rl = (1.0 - gl) / gl
            rl = np.where(gl > 1e-14, rl, 1e18)
            rl = np.where(np.isfinite(rl), rl, 1e18)
            local_max = float(np.max(rl))
            if local_max > best_val:
                best_val = local_max
        
        # L-BFGS-B on exact objective
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-13, "maxls": 50},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 5: Adaptive fine grid polish
    if remaining() > 5.0:
        for step in (0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002):
            if remaining() < 2.0:
                break
            for i in range(-4, 5):
                for j in range(-4, 5):
                    for k in range(-4, 5):
                        if remaining() < 0.5:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
    
    # Phase 6: Final ultra-fine refinement
    if remaining() > 2.0:
        for step in (0.00001, 0.000005, 0.000002, 0.000001):
            if remaining() < 1.0:
                break
            for i in range(-3, 4):
                for j in range(-3, 4):
                    for k in range(-3, 4):
                        if remaining() < 0.3:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
    
    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
