"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 10395

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n, highest degree first."""
    # H_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * n! / (k! (n-2k)!) * (2x)^{n-2k}
    coeffs = {}
    for k in range(n // 2 + 1):
        coeff = (-1)**k * float(factorial(n)) / (float(factorial(k)) * float(factorial(n - 2*k))) * (2**(n - 2*k))
        coeffs[n - 2*k] = coeff
    # Convert to list from degree n down to 0
    result = [0.0] * (n + 1)
    for deg, c in coeffs.items():
        result[n - deg] = c
    return result

_H0 = _hermite_coeffs(0)   # [1]
_H4 = _hermite_coeffs(4)   # [1, 0, -6, 0, 3]
_H8 = _hermite_coeffs(8)   # [1, 0, -28, 0, 210, 0, -420, 0, 105]
_H12 = _hermite_coeffs(12) # [1, 0, -84, 0, 2520, 0, -30240, 0, 151200, 0, -221760, 0, 10395]

def _build_q_coeffs(c0, c1, c2):
    """
    Build coefficients of q(x) = g(x)/x^2 where g = c0*H0 + c1*H4 + c2*H8 + c3*H12,
    c3 chosen so g(0) = 0. q must divide exactly (g has factor x^2).
    Returns list of coefficients of q from highest degree to lowest.
    """
    H0_0 = _H0[-1]  # H0(0) = 1
    H4_0 = _H4[-1]  # H4(0) = 3
    H8_0 = _H8[-1]  # H8(0) = 105
    H12_0 = _H12[-1]  # H12(0) = 10395
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Degrees: H0 is deg 0, H4 is deg 4, H8 is deg 8, H12 is deg 12
    # g has degree 12 (if c3 != 0)
    # g(x)/x^2 has degree 10
    
    # Build g coefficients (degree 12 down to 0)
    g = [0.0] * 13
    for i, c in enumerate(_H0):
        g[12 - i] += c0 * c  # H0 has 1 term, degree 0, so g[12] += c0*1... wait
    # Actually _H0 = [1.0] (degree 0), so it contributes to g[12 - 0] = g[12]? No.
    # _H0 has length 1, index 0 corresponds to degree 0.
    # Let me redo: _Hn is a list of length n+1, where index i corresponds to degree (n-i).
    # So _H0 = [1.0] means degree 0 coefficient is 1.0
    # _H4 = [1, 0, -6, 0, 3] means deg4=1, deg3=0, deg2=-6, deg1=0, deg0=3
    # _H8 = [1, 0, -28, 0, 210, 0, -420, 0, 105]
    # _H12 = [1, 0, -84, 0, 2520, 0, -30240, 0, 151200, 0, -221760, 0, 10395]
    
    # g(x) degree 12 down to 0
    g = [0.0] * 13
    # H0 contributes to degree 0
    g[12] += c0 * _H0[0]  # Wait, _H0[0] is the coefficient of x^0 in H0, which is 1
    # No, g[deg] should be coefficient of x^deg. Let me use a dict approach.
    
    g_coeffs = {d: 0.0 for d in range(13)}
    # H0: only degree 0
    g_coeffs[0] += c0 * _H0[-1]  # _H0[-1] = coeff of x^0
    # H4: degrees 4, 2, 0
    for i, c in enumerate(_H4):
        deg = 4 - i
        g_coeffs[deg] += c1 * c
    # H8: degrees 8, 6, 4, 2, 0
    for i, c in enumerate(_H8):
        deg = 8 - i
        g_coeffs[deg] += c2 * c
    # H12: degrees 12, 10, 8, 6, 4, 2, 0
    for i, c in enumerate(_H12):
        deg = 12 - i
        g_coeffs[deg] += c3 * c
    
    # q(x) = g(x)/x^2, so q has degrees 10 down to 0
    # q_coeffs[deg] = g_coeffs[deg+2] for deg 0..10
    # Check: g_coeffs[0] and g_coeffs[1] should be 0 for exact division
    # g_coeffs[0] = c0*1 + c1*3 + c2*105 + c3*10395 = 0 by construction
    # g_coeffs[1] = 0 (all Hermite polys have only even powers)
    
    q_coeffs = [0.0] * 11  # degrees 10 down to 0
    for deg in range(11):
        q_coeffs[deg] = g_coeffs[deg + 2]
    
    return q_coeffs

def _largest_sign_changing_root(q_coeffs):
    """
    Find the largest real root of q where q changes sign.
    q_coeffs is from highest degree to lowest.
    """
    # Find all real roots
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return None
    
    # For each real root, check if q changes sign
    # q changes sign at root r if q(r-eps) * q(r+eps) < 0
    best_r = None
    eps = 1e-8
    
    for r in real_roots:
        # Evaluate q at r - eps and r + eps
        q_minus = np.polyval(q_coeffs, r - eps)
        q_plus = np.polyval(q_coeffs, r + eps)
        
        if q_minus * q_plus < 0:
            # Sign change at r
            if best_r is None or r > best_r:
                best_r = r
    
    return best_r

def _compute_value(c0, c1, c2):
    """Compute the value r^2 / (2*pi) for given coefficients."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    r = _largest_sign_changing_root(q_coeffs)
    if r is None:
        return float('inf')
    return r**2 / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Start with seed
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    best_value = _compute_value(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Use differential evolution for global search
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    def objective(x):
        try:
            v = _compute_value(x[0], x[1], x[2])
            if v == float('inf'):
                return 1e6
            return v
        except:
            return 1e6
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, tol=1e-8, 
            seed=42, polish=True, workers=1
        )
        if result.success and result.fun < best_value:
            best_coeffs = result.x
            best_value = result.fun
    except:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return np.array(best_coeffs)
    
    # Local refinement with Nelder-Mead
    try:
        local_result = minimize(
            objective, best_coeffs, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if local_result.fun < best_value:
            best_coeffs = local_result.x
            best_value = local_result.fun
    except:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return np.array(best_coeffs)
    
    # Try a few more local optimizations from perturbed starting points
    rng = np.random.RandomState(123)
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        perturbation = rng.normal(0, 0.1, size=3)
        start = best_coeffs + perturbation
        try:
            local_result = minimize(
                objective, start, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if local_result.fun < best_value:
                best_coeffs = local_result.x
                best_value = local_result.fun
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
