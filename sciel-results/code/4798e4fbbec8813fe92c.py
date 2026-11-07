"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brentq
import time

def _compute_q_coeffs(c0, c1, c2):
    """Compute coefficients of q(x) = g(x)/x^2 as a polynomial in x (degree 10).
    q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    (only even powers since g is even)
    """
    c3 = -(c0 + 3*c1 + 15*c2) / 105.0
    
    # q(x) = c3*(x^10 - 60x^8 + 900x^6 - 3600x^4 + 2880x^2 - 432)
    #       + c2*(x^6 - 24x^4 + 72x^2 - 32)
    #       + c1*(x^2 - 6)
    a10 = c3
    a8 = -60*c3
    a6 = 900*c3 + c2
    a4 = -3600*c3 - 24*c2
    a2 = 2880*c3 + 72*c2 + c1
    a0 = -432*c3 - 32*c2 - 6*c1
    
    return np.array([a10, a8, a6, a4, a2, a0])  # coefficients for x^10, x^8, x^6, x^4, x^2, x^0

def _find_largest_sign_changing_root(coeffs):
    """Find the largest real root where q changes sign.
    coeffs: [a10, a8, a6, a4, a2, a0] for q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    """
    # Build full polynomial for numpy (descending powers)
    # q(x) = a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    full_coeffs = np.array([coeffs[0], 0, coeffs[1], 0, coeffs[2], 0, coeffs[3], 0, coeffs[4], 0, coeffs[5]])
    
    roots = np.roots(full_coeffs)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    real_roots.sort(reverse=True)
    
    # Find largest root where sign changes
    # Evaluate q at points near each root to check sign change
    for r in real_roots:
        # Check sign change by evaluating at r - eps and r + eps
        eps = 1e-8 * max(1, abs(r))
        q_minus = _eval_q(r - eps, coeffs)
        q_plus = _eval_q(r + eps, coeffs)
        if q_minus * q_plus < 0:
            return r
    return None

def _eval_q(x, coeffs):
    """Evaluate q(x) given coefficients [a10, a8, a6, a4, a2, a0]"""
    a10, a8, a6, a4, a2, a0 = coeffs
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    x8 = x4*x4
    x10 = x8*x2
    return a10*x10 + a8*x8 + a6*x6 + a4*x4 + a2*x2 + a0

def _objective(params):
    """Objective: minimize r^2 where r is largest sign-changing root of q"""
    c0, c1, c2 = params
    coeffs = _compute_q_coeffs(c0, c1, c2)
    r = _find_largest_sign_changing_root(coeffs)
    if r is None:
        return 1e10
    return r*r

def construct():
    start_time = time.time()
    deadline = 200.0  # safety margin before 210s limit
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_params = seed.copy()
    best_value = _objective(seed)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 50
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            x0 = seed
        else:
            # Random perturbation around seed or uniform in reasonable range
            if i < n_starts // 2:
                x0 = seed + rng.randn(3) * 0.1
            else:
                x0 = rng.uniform([-2, -2, -2], [2, 2, 2])
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
