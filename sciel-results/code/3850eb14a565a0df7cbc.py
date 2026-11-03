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
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # Precompute (k+1)!
    # To avoid overflow in factorial for large k, we can use logs?
    # But poly coefficients might be huge.
    # Let's stick to direct calculation if possible, or scale.
    # For now, direct calculation.
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
        return 0.0 # Large positive value to stop optimization? No, return inf
    
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
    
    # Phase 1: Explore large scales
    # The norm grows with root magnitude. Try scaling up.
    scales = [4.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    for s in scales:
        if _time_left() < 60:
            break
        roots = [s * (i + 1) for i in range(n_roots)]
        val = _c4_bound(roots, m)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_roots = roots[:]
            
    # Phase 2: Local optimization to fine-tune roots
    # We want to maximize norm_sq, so minimize -log(norm_sq)
    if _time_left() > 120:
        # Start from the best roots found
        x0 = np.array(best_roots)
        
        # Use Nelder-Mead for local search
        # It doesn't require gradients, which is good for potential non-smoothness
        # or numerical noise.
        
        # We need to ensure roots remain positive.
        # Nelder-Mead can go negative. We can penalize negative roots in the objective.
        # Or use bounds? Nelder-Mead doesn't support bounds directly.
        # We can transform variables: r_i = exp(y_i).
        # Then y_i can be any real number.
        
        def _neg_log_norm_transformed(y, m=12):
            roots = np.exp(y)
            return _neg_log_norm(roots, m)
            
        y0 = np.log(x0)
        
        # Run optimization
        # Maximize the number of iterations within time limit
        # Nelder-Mead is relatively fast.
        
        try:
            # Use a reasonable number of iterations
            # Each evaluation takes ~1ms. 600s allows ~600,000 evals.
            # Nelder-Mead usually converges in fewer iterations.
            # Let's set a generous maxiter.
            result = minimize(
                _neg_log_norm_transformed, 
                y0, 
                method='Nelder-Mead', 
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < _neg_log_norm_transformed(y0):
                best_roots = np.exp(result.x)
                best_val = _c4_bound(best_roots, m)
                
        except Exception as e:
            pass
            
    # Ensure we return valid positive distinct roots
    if len(best_roots) != n_roots or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
        
    return best_roots
# EVOLVE-BLOCK-END
