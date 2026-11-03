"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """
    Compute coefficients of L_n^(alpha)(x) in ascending power of x.
    L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * C(n+alpha, n-k) * x^k / k!
    """
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        # C(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
        try:
            from math import lgamma
            log_val = (lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_norm_sq(poly):
    """
    Compute ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx
    = sum_{i,j} poly[i] poly[j] Gamma(i+j+1+1)
    = sum_k (sum_i poly[i] poly[k-i]) * (k+1)!
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # Precompute (k+1)! for k = 0..max_k
    fact = np.ones(max_k + 2)
    for j in range(1, max_k + 2):
        fact[j] = fact[j - 1] * j
    weights = fact[1:max_k + 2]
    norm_sq = np.dot(sq, weights)
    return norm_sq


def _c4_bound(roots, m=12):
    """
    Compute upper bound on C4 for given roots.
    psi(x) = L_m^(1)(x) * prod_i (x - r_i)^2
    C4 <= 1 / ||psi||^2
    """
    if _time_left() < 5:
        return float('inf')
    
    # Get L_m^(1)(x) coefficients
    coeff_L = _laguerre_coeffs(m, 1)
    
    # Multiply by (x - r)^2 for each root
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    # Check for numerical issues
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq


def _make_roots(scale, alpha, n_roots=12):
    """Generate roots as scale * (i+1)^alpha for i=0..n_roots-1"""
    return [scale * ((i + 1) ** alpha) for i in range(n_roots)]


def construct():
    m = 12
    n_roots = 12
    
    # Baseline: the seed roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0  # fallback
    
    best_roots = seed_roots[:]
    best_val = seed_val

    # Coarse grid over scale and exponent
    scales = np.linspace(0.5, 25.0, 60)
    alphas = np.linspace(0.25, 1.15, 45)
    
    for s in scales:
        if _time_left() < 60:
            break
        for a in alphas:
            if _time_left() < 60:
                break
            roots = _make_roots(s, a)
            if min(roots) <= 0 or len(set(round(r, 9) for r in roots)) != 12:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_roots = roots[:]

    # Local refinement using scipy.optimize.minimize around the best point
    if best_val != float('inf') and _time_left() > 90:
        # Recover best s, a from best_roots
        r0 = best_roots[0]
        if r0 > 0 and best_roots[1] > 0:
            ratio = best_roots[1] / best_roots[0]
            if ratio > 1:
                alpha_est = math.log(ratio) / math.log(2.0)
            else:
                alpha_est = 0.5
            scale_est = r0
            
            # Define objective function for minimization
            def objective(params):
                s, a = params
                if s <= 0 or a < 0 or a > 2:
                    return 1e10
                roots = _make_roots(s, a)
                if min(roots) <= 0:
                    return 1e10
                val = _c4_bound(roots, m)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            # Try several local optimizations from different starting points
            start_points = [
                [scale_est, alpha_est],
                [scale_est * 0.5, alpha_est],
                [scale_est * 2.0, alpha_est],
                [scale_est, alpha_est * 0.5],
                [scale_est, alpha_est * 2.0],
            ]
            
            for sp in start_points:
                if _time_left() < 90:
                    break
                try:
                    result = minimize(objective, sp, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.success or result.fun < best_val:
                        s_opt, a_opt = result.x
                        if s_opt > 0 and a_opt > 0 and a_opt < 2:
                            roots_opt = _make_roots(s_opt, a_opt)
                            if min(roots_opt) > 0 and len(set(round(r, 9) for r in roots_opt)) == 12:
                                val_opt = _c4_bound(roots_opt, m)
                                if np.isfinite(val_opt) and val_opt < best_val:
                                    best_val = val_opt
                                    best_roots = roots_opt[:]
                except Exception:
                    pass

    return best_roots
# EVOLVE-BLOCK-END
