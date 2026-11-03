"""Refined Cohn-Goncalves Laguerre construction: Direct root optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize

_DEADLINE = 600.0
_START = time.time()

def _time_left():
    return _DEADLINE - time.time()

def _compute_c4_bound(roots, m=12):
    """
    Compute the upper bound on C4 for given roots.
    C4 <= 1 / ||psi||^2 where psi(x) = L_m^(1)(x) * prod (x - r_i)^2
    and ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx.
    """
    if _time_left() < 5:
        return float('inf')
    
    # Build the polynomial psi(x)
    # 1. Get coefficients of L_m^(1)(x)
    # Using roots_genlaguerre to get zeros, but we need the polynomial itself.
    # scipy.special.eval_genlaguerre gives values, not coeffs directly.
    # We can reconstruct coeffs from roots or use a recurrence.
    # For n=12, we can just sample and fit, or use the explicit formula.
    # Let's use the explicit formula for coefficients of L_n^(alpha)(x).
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
    
    n = m
    alpha = 1.0
    coeff_L = np.zeros(n + 1)
    for k in range(n + 1):
        # binom(n+alpha, n-k)
        # Use math.comb for integers, but alpha is 1.0 so n+alpha is integer.
        # n-k is integer.
        try:
            from math import comb
            binom_val = comb(int(n + alpha), int(n - k))
        except:
            # Fallback for non-integer if needed, but here alpha=1
            binom_val = 1.0
        coeff_L[k] = ((-1) ** k) * binom_val / math.factorial(k)
        
    # coeff_L is in ascending power of x.
    # L_m^(1)(x) = c0 + c1 x + ... + cn x^n
    
    # 2. Multiply by (x - r_i)^2 for each root
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        # (x - r)^2 = x^2 - 2r x + r^2
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
        
    # 3. Compute the norm squared
    # ||psi||^2 = sum_{i,j} poly[i] poly[j] Gamma(i+j+1+1)
    # = sum_{i,j} poly[i] poly[j] (i+j+1)!
    # Let k = i+j. sum_k (sum_i poly[i] poly[k-i]) * (k+1)!
    
    sq = np.convolve(poly, poly)
    # sq[k] = sum_{i=0}^k poly[i] poly[k-i]
    # We need (k+1)! for k from 0 to 2*deg
    deg = len(poly) - 1
    max_k = 2 * deg
    
    # Precompute factorials up to max_k + 1
    fact = np.ones(max_k + 1)
    for k in range(1, max_k + 1):
        fact[k] = fact[k - 1] * k
        
    # (k+1)! is fact[k+1]
    weights = fact[1:max_k + 2] # indices 1 to max_k+1
    
    norm_sq = np.dot(sq, weights)
    
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
        
    return 1.0 / norm_sq

def construct():
    m = 12
    
    # Initial guess: Zeros of L_12^(1)(x)
    # The optimal roots are often close to the zeros of the Laguerre polynomial.
    # We scale them to ensure positivity and spread.
    x, w = roots_genlaguerre(m, 1)
    
    # The roots of L_n^(1) are positive.
    # Let's start with these, perhaps scaled by a factor.
    # The scale factor is also a degree of freedom.
    # We can optimize the 12 roots directly.
    
    # Sort roots to maintain order for stability in optimization
    x0 = np.sort(x)
    
    # Define objective function for optimizer
    def objective(params):
        # params are the 12 roots
        # Ensure positivity by exp transformation or bounds
        # L-BFGS-B supports bounds
        roots = params
        if np.any(roots <= 0):
            return float('inf')
        return _compute_c4_bound(roots, m)
    
    # Bounds: roots must be positive.
    # Upper bound: unlikely to be huge. 1000 is safe.
    bounds = [(1e-6, 1000.0)] * m
    
    # Run optimization
    best_roots = x0
    best_val = objective(best_roots)
    
    # Try a few different initial scales to avoid local minima
    scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for scale in scales:
        if _time_left() < 30:
            break
            
        init_roots = x0 * scale
        # Ensure distinct and positive
        if np.any(init_roots <= 0):
            continue
            
        try:
            res = minimize(
                objective, 
                init_roots, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success and np.isfinite(res.fun):
                if res.fun < best_val:
                    best_val = res.fun
                    best_roots = res.x
        except Exception:
            pass

    # Final polish with tighter tolerances if time allows
    if _time_left() > 10 and best_val < float('inf'):
        try:
            res = minimize(
                objective, 
                best_roots, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success and np.isfinite(res.fun):
                if res.fun < best_val:
                    best_val = res.fun
                    best_roots = res.x
        except Exception:
            pass

    # Ensure the returned roots are a list of floats
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
