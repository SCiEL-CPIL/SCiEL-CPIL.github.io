# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

# Hermite polynomial values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# Coefficients for Hermite polynomials H_n(x) for even n up to 12
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
# H_12(x) = 4096x^12 - 24576x^10 + 430080x^8 - 2457600x^6 + 6220800x^4 - 5443200x^2 + 665280

def _get_poly_coeffs(a0, a4, a8):
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    # Coefficients for x^12, x^10, x^8, x^6, x^4, x^2, x^0
    c12 = 4096.0 * a12
    c10 = -24576.0 * a12 + 256.0 * a8
    c8 = 430080.0 * a12 - 3072.0 * a8 + 16.0 * a4
    c6 = -2457600.0 * a12 + 10752.0 * a8 - 48.0 * a4
    c4 = 6220800.0 * a12 - 10752.0 * a8 + 12.0 * a4 + a0
    c2 = -5443200.0 * a12 + 1680.0 * a8 - 48.0 * a4
    c0 = 665280.0 * a12 + 1680.0 * a8 + 12.0 * a4 + a0
    
    # Construct full coefficient array for np.roots (descending powers)
    # x^12, x^11, x^10, x^9, x^8, x^7, x^6, x^5, x^4, x^3, x^2, x^1, x^0
    coeffs = np.zeros(13)
    coeffs[0] = c12
    coeffs[2] = c10
    coeffs[4] = c8
    coeffs[6] = c6
    coeffs[8] = c4
    coeffs[10] = c2
    coeffs[12] = c0
    return coeffs, a12

def _objective(c):
    a0, a4, a8 = c
    coeffs, a12 = _get_poly_coeffs(a0, a4, a8)
    
    # Find roots of P(x)
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e30
        
    # Keep only real roots within a reasonable range
    # e^{-x^2/2} is negligible beyond |x| > 10
    real_roots = []
    for r in roots:
        if np.isreal(r):
            r_real = float(np.real(r))
            if -10.0 <= r_real <= 10.0:
                real_roots.append(r_real)
                
    # Sort roots
    real_roots.sort()
    
    # Define the function to integrate
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # We can evaluate P(x) using Horner's method with the coeffs for stability
    # Or just sum the terms. Horner is better.
    # coeffs are for x^12 ... x^0
    
    def integrand(x):
        # Evaluate P(x)
        # Using coeffs
        # c12*x^12 + c10*x^10 + ...
        # Horner for even powers only? 
        # Let y = x^2
        # P(x) = c12*y^6 + c10*y^5 + c8*y^4 + c6*y^3 + c4*y^2 + c2*y + c0
        y = x * x
        p_val = c12
        p_val = p_val * y + c10
        p_val = p_val * y + c8
        p_val = p_val * y + c6
        p_val = p_val * y + c4
        p_val = p_val * y + c2
        p_val = p_val * y + c0
        
        g_val = p_val * np.exp(-0.5 * y)
        return abs(g_val)
        
    # Integrate
    if not real_roots:
        try:
            L1, err = quad(integrand, -10.0, 10.0, limit=100)
        except Exception:
            return 1e30
    else:
        # Split integration at roots
        points = real_roots
        try:
            L1, err = quad(integrand, -10.0, 10.0, points=points, limit=200)
        except Exception:
            # Fallback to simple quad if points cause issues
            try:
                L1, err = quad(integrand, -10.0, 10.0, limit=200)
            except Exception:
                return 1e30
                
    if not np.isfinite(L1) or L1 < 0:
        return 1e30
        
    return L1 * L1

def construct():
    t0 = time.time()
    deadline = t0 + 180.0 # Keep margin for 210s limit
    
    best_c = _SEED.copy()
    best_val = _objective(best_c)
    
    # Global search
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    if time.time() < deadline - 60.0:
        try:
            res_global = differential_evolution(_objective, bounds, seed=42, maxiter=100, popsize=15, tol=1e-8, polish=False)
            if res_global.fun < best_val:
                best_val = float(res_global.fun)
                best_c = np.asarray(res_global.x, dtype=float).copy()
        except Exception:
            pass
            
    # Local polish
    if time.time() < deadline - 20.0:
        try:
            res_local = minimize(_objective, best_c, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
            if res_local.fun < best_val:
                best_val = float(res_local.fun)
                best_c = np.asarray(res_local.x, dtype=float).copy()
        except Exception:
            pass
            
    # Another polish from seed in case global went to a bad local min?
    # Or just polish the current best again with a different method
    if time.time() < deadline - 5.0:
        try:
            res_powell = minimize(_objective, best_c, method='Powell', options={'maxiter': 500, 'ftol': 1e-15})
            if res_powell.fun < best_val:
                best_val = float(res_powell.fun)
                best_c = np.asarray(res_powell.x, dtype=float).copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_c)):
        return _SEED.copy()
        
    return best_c
# EVOLVE-BLOCK-END
