"""Tuned Fourier-invariant Hermite-Gaussian test function.
Minimizes the upper bound max_{lambda>=0} (1-g(lambda))/g(lambda)
over three free Hermite coefficients (degrees 0, 4, 8), with degree-12
fixed so g(0)=0. Uses smooth log-sum-exp surrogate for optimization,
then exact min-max polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

# Physicists' Hermite polynomials evaluated at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 798336.0

def _hermite_at(x):
    """Evaluate H0, H4, H8, H12 (physicists') at points x (array or scalar)."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h4 = 16.0*x**4 - 48.0*x**2 + 12.0
    h8 = 256.0*x**8 - 3584.0*x**6 + 13440.0*x**4 - 13440.0*x**2 + 1680.0
    h12 = (4096.0*x**12 - 212992.0*x**10 + 3317760.0*x**8
           - 25333440.0*x**6 + 100810080.0*x**4 - 201620160.0*x**2 + 798336.0)
    return h0, h4, h8, h12

def _g_of_lambda(a, lam):
    """g(lam) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)(lam) * exp(-lam^2/2)
    with a12 fixed so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 798336.0
    h0, h4, h8, h12 = _hermite_at(lam)
    poly = a0*h0 + a4*h4 + a8*h8 + a12*h12
    return poly * np.exp(-0.5*lam*lam)

def _exact_bound(a, n_coarse=50000, n_fine=20000):
    """Compute max_{lam>=0} (1-g(lam))/g(lam) on a dense grid with local refinement."""
    lam = np.concatenate([
        np.linspace(0.0, 40.0, n_coarse),
        np.geomspace(1e-8, 80.0, 5000)
    ])
    g = _g_of_lambda(a, lam)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(lam[idx])
    if 0.0 < x0 < 80.0:
        local = np.linspace(max(0.0, x0 - 0.1), x0 + 0.1, n_fine)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide='ignore', invalid='ignore'):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        val = max(val, float(np.max(rl)))
    return val

def _smooth_bound(a, n_grid=20000, beta=50.0):
    """Smooth surrogate: log-sum-exp approximation to max (1-g)/g."""
    lam = np.concatenate([
        np.linspace(0.0, 40.0, n_grid),
        np.geomspace(1e-8, 80.0, 2000)
    ])
    g = _g_of_lambda(a, lam)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    # log-sum-exp for smooth max
    m = np.max(ratio)
    if m > 1e17:
        return m
    s = np.sum(np.exp(ratio - m))
    return m + np.log(s) / beta

def _objective(a):
    """Primary objective: smooth surrogate for min-max bound."""
    val = _smooth_bound(a)
    if not np.isfinite(val):
        return 1e18
    return val

def _objective_exact(a):
    """Exact min-max bound for final polish."""
    val = _exact_bound(a)
    if not np.isfinite(val):
        return 1e18
    return val

def construct():
    seed_result = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_result = seed_result.copy()
    best_bound = _objective_exact(seed_result)
    
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    # Phase 1: Global search with differential evolution using smooth surrogate
    if time.time() < _DEADLINE - 100:
        try:
            de_res = differential_evolution(
                _objective, bounds, seed=42, maxiter=200, tol=1e-10,
                polish=False, init='sobol', popsize=25,
                mutation=(0.4, 1.0), recombination=0.8
            )
            if de_res.fun < best_bound:
                best_bound = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead with smooth surrogate
    starts = [
        seed_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-0.7, 0.025, 0.00015]),
        np.array([-1.3, 0.055, 0.00045]),
        best_result.copy(),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(
                _objective, start, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polish with smooth surrogate
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                _objective, best_result, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 200}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Exact min-max polish with tighter grid
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _objective_exact, best_result, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
