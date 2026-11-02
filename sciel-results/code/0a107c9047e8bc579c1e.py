import numpy as np
import time
import math
from scipy.optimize import minimize

# Precompute grid for fast evaluation
_x_grid = np.linspace(-8, 8, 4001)
_dx = _x_grid[1] - _x_grid[0]
_gauss = np.exp(-_x_grid**2 / 2)

# Probabilists' Hermite polynomials precomputed on grid
def _he0(x): return np.ones_like(x)
def _he4(x): return x**4 - 6*x**2 + 3
def _he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def _he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

_H0 = _he0(_x_grid)
_H4 = _he4(_x_grid)
_H8 = _he8(_x_grid)
_H12 = _he12(_x_grid)

# Values at 0 for the constraint g(0)=0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 10395.0

def _c3_from(c0, c1, c2):
    return -(c0 + _H4_0*c1 + _H8_0*c2) / _H12_0

def _g_values(coeffs):
    """Compute g(x) on the precomputed grid."""
    c0, c1, c2 = coeffs
    c3 = _c3_from(c0, c1, c2)
    return (c0*_H0 + c1*_H4 + c2*_H8 + c3*_H12) * _gauss

def _exact_bound(coeffs):
    """Exact bound: sqrt(2*pi) * L1 / L_inf for Fourier-invariant g."""
    g = _g_values(coeffs)
    l1 = np.sum(np.abs(g)) * _dx
    linf = np.max(np.abs(g))
    if linf < 1e-15:
        return 1e10
    return math.sqrt(2 * math.pi) * l1 / linf

def _smooth_bound(coeffs, temperature=0.005):
    """Smooth surrogate: sqrt(2*pi) * L1 / soft_min(|g|).
    
    Uses log-sum-exp for a smooth approximation of min(|g|),
    which is the non-differentiable part of the exact objective.
    """
    g = _g_values(coeffs)
    l1 = np.sum(np.abs(g)) * _dx
    
    # Soft minimum of |g| using log-sum-exp
    abs_g = np.abs(g)
    # Scale for numerical stability
    m = np.max(abs_g)
    if m < 1e-15:
        return 1e10
    scaled = abs_g / m
    # log-sum-exp of -scaled/temperature gives -soft_min/temperature
    log_sum_exp = np.log(np.sum(np.exp(-scaled / temperature)))
    soft_min = -temperature * (log_sum_exp - np.log(len(scaled))) * m
    
    if soft_min < 1e-15:
        return 1e10
    return math.sqrt(2 * math.pi) * l1 / soft_min

def _smooth_bound_v2(coeffs, temperature=0.002):
    """Alternative smooth surrogate using power-mean approximation.
    
    p-norm with large p approximates min. As p -> infinity, ||.||_p / n^(1/p) -> min.
    Here we use a moderate p for smoothness.
    """
    g = _g_values(coeffs)
    l1 = np.sum(np.abs(g)) * _dx
    abs_g = np.abs(g)
    
    # Use p-norm approximation to min: (sum |g|^p)^(1/p) / n^(1/p) -> min as p->inf
    # For p=200, this is close to min but smooth
    p = 200
    # Numerical stability: scale first
    m = np.max(abs_g)
    if m < 1e-15:
        return 1e10
    scaled = abs_g / m
    # sum(scaled^p)^(1/p) * m / n^(1/p)
    sum_p = np.sum(scaled**p)
    if sum_p < 1e-300:
        return 1e10
    pnorm_min = (sum_p ** (1.0/p)) * m / (len(scaled) ** (1.0/p))
    
    if pnorm_min < 1e-15:
        return 1e10
    return math.sqrt(2 * math.pi) * l1 / pnorm_min

def construct():
    t0 = time.time()
    deadline = t0 + 175  # safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = _exact_bound(best)
    
    # Phase 1: Local refinement from seed using smooth surrogate
    if time.time() < deadline - 60:
        try:
            res = minimize(_smooth_bound, seed, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline - 30 and res.fun < best_val:
                # Verify with exact bound
                exact_v = _exact_bound(res.x)
                if exact_v < best_val:
                    best = res.x.copy()
                    best_val = exact_v
        except Exception:
            pass
    
    # Phase 2: Multi-start with diverse initial points using smooth surrogate
    if time.time() < deadline - 45:
        rng = np.random.default_rng(42)
        # Try points in a wider range, biased toward seed region
        n_starts = 20
        for i in range(n_starts):
            if time.time() > deadline - 30:
                break
            # Mix of local perturbations and wider exploration
            if i < 10:
                start = seed + rng.normal(0, 0.05, size=3)
            else:
                start = np.array([rng.uniform(-2.0, -0.5), 
                                  rng.uniform(-0.2, 0.2), 
                                  rng.uniform(-0.005, 0.005)])
            try:
                res = minimize(_smooth_bound_v2, start, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-11, 'adaptive': True})
                if time.time() > deadline - 30:
                    break
                if res.fun < best_val:
                    exact_v = _exact_bound(res.x)
                    if exact_v < best_val:
                        best = res.x.copy()
                        best_val = exact_v
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish using exact objective
    if time.time() < deadline - 15:
        try:
            res = minimize(_exact_bound, best, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 4: One more ultra-fine polish if time permits
    if time.time() < deadline - 5:
        try:
            res = minimize(_exact_bound, best, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
