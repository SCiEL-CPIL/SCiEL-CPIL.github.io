"""Refined Cohn-Goncalves Laguerre construction: Optimized root search."""
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
    """
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = (math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff

def _compute_norm_sq(poly):
    """
    Compute ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    fact = np.ones(max_k + 2)
    for j in range(1, max_k + 2):
        fact[j] = fact[j - 1] * j
    weights = fact[1:max_k + 2]
    norm_sq = np.dot(sq, weights)
    return norm_sq

def _c4_bound(roots, m=12):
    """
    Compute upper bound on C4 for given roots.
    Returns the bound value (lower is better).
    """
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq

def _neg_log_norm(roots, m=12):
    """
    Objective function to minimize: -log(norm_sq).
    Equivalent to maximizing norm_sq.
    """
    if _time_left() < 2:
        return 1e100
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return 1e100
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return 1e100
        
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e100
        
    return -math.log(norm_sq)

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
        seed_val = 1.0
        
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Coarse grid over scale and exponent
    # The norm grows with root magnitude and depends on spacing
    scales = np.linspace(1.0, 50.0, 40)
    alphas = np.linspace(0.3, 1.3, 35)
    
    for s in scales:
        if _time_left() < 120:
            break
        for a in alphas:
            if _time_left() < 120:
                break
            roots = _make_roots(s, a)
            if min(roots) <= 0:
                continue
            # Check distinctness
            if len(set(round(r, 9) for r in roots)) != 12:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_roots = roots[:]
    
    # Phase 2: Local refinement around best (concentric rings)
    if best_val != float('inf') and _time_left() > 180:
        r0 = best_roots[0]
        if r0 > 0 and best_roots[1] > 0:
            ratio = best_roots[1] / best_roots[0]
            if ratio > 1.0:
                alpha_est = math.log(ratio) / math.log(2.0)
            else:
                alpha_est = 1.0
            scale_est = r0
            
            for ds in (-2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
                for da in (-0.3, -0.15, -0.07, 0.0, 0.07, 0.15, 0.3):
                    if _time_left() < 120:
                        break
                    s = scale_est * (1.0 + ds)
                    a = alpha_est + da
                    if s <= 0:
                        continue
                    roots = _make_roots(s, a)
                    if min(roots) <= 0:
                        continue
                    if len(set(round(r, 9) for r in roots)) != 12:
                        continue
                    val = _c4_bound(roots, m)
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best_roots = roots[:]
    
    # Phase 3: Nelder-Mead local optimization in log-space
    if _time_left() > 60:
        x0 = np.array(best_roots)
        if all(r > 0 for r in x0):
            def _neg_log_norm_transformed(y, m=12):
                roots = np.exp(y)
                return _neg_log_norm(roots, m)
                
            y0 = np.log(x0)
            
            try:
                result = minimize(
                    _neg_log_norm_transformed, 
                    y0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                if result.success or (np.isfinite(result.fun) and result.fun < _neg_log_norm_transformed(y0)):
                    candidate_roots = np.exp(result.x)
                    if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                        cand_val = _c4_bound(candidate_roots, m)
                        if np.isfinite(cand_val) and cand_val < best_val:
                            best_val = cand_val
                            best_roots = candidate_roots.tolist()
                            
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with tighter tolerance
    if _time_left() > 30:
        x0 = np.array(best_roots)
        if all(r > 0 for r in x0):
            def _neg_log_norm_transformed(y, m=12):
                roots = np.exp(y)
                return _neg_log_norm(roots, m)
                
            y0 = np.log(x0)
            
            try:
                result = minimize(
                    _neg_log_norm_transformed, 
                    y0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                if result.success or (np.isfinite(result.fun) and result.fun < _neg_log_norm_transformed(y0)):
                    candidate_roots = np.exp(result.x)
                    if all(r > 0 for r in candidate_roots) and len(set(round(r, 9) for r in candidate_roots)) == 12:
                        cand_val = _c4_bound(candidate_roots, m)
                        if np.isfinite(cand_val) and cand_val < best_val:
                            best_val = cand_val
                            best_roots = candidate_roots.tolist()
                            
            except Exception:
                pass
            
    # Ensure we return valid positive distinct roots
    if len(best_roots) != n_roots or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
        
    return best_roots
# EVOLVE-BLOCK-END
