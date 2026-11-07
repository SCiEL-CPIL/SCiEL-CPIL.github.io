"""Rewritten program: multi-start Nelder-Mead + polish for uncertainty inequality."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, root_scalar
from fractions import Fraction

_DEADLINE = 180.0  # safety margin before 210s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE

def _build_q_coeffs(c0, c1, c2):
    """
    g = c0*H0 + c1*H4 + c2*H8 + c3*H12, c3 = -(c0*H0(0)+c1*H4(0)+c2*H8(0))/H12(0)
    H0=1, H4(0)=3, H8(0)=15, H12(0)=105
    c3 = -(c0 + 3*c1 + 15*c2)/105
    
    g(x) = c0 + c1(x^4-6x^2+3) + c2(x^8-24x^6+72x^4-32x^2+15) + c3(x^12-60x^10+660x^8-3600x^6+5040x^4-1680x^2+105)
    
    q = g/x^2 (exact division since g(0)=0)
    q(x) = c3*x^10 - 60*c3*x^8 + (660*c3+c2)*x^6 + (-3600*c3-24*c2+c1)*x^4 + (5040*c3+72*c2-6*c1)*x^2 + (-1680*c3-32*c2+3*c1)
    
    Wait, let me redo. g(x)/x^2:
    g(x) = c3*x^12 + (660*c3+c2)*x^8 + (-60*c3-24*c2+c1)*x^6 + (5040*c3+72*c2-6*c1)*x^4 + (-1680*c3-32*c2+3*c1)*x^2 + (c0+3*c1+15*c2+105*c3)
    
    Since c3 = -(c0+3*c1+15*c2)/105, the constant term is 0. Good.
    
    q(x) = g(x)/x^2 = c3*x^10 + (660*c3+c2)*x^6 + (-60*c3-24*c2+c1)*x^4 + (5040*c3+72*c2-6*c1)*x^2 + (-1680*c3-32*c2+3*c1)
    
    Let y = x^2. Then q as function of y:
    Q(y) = c3*y^5 + (660*c3+c2)*y^3 + (-60*c3-24*c2+c1)*y^2 + (5040*c3+72*c2-6*c1)*y + (-1680*c3-32*c2+3*c1)
    
    This is a quintic in y. We need the largest real root y* where Q changes sign (i.e., Q crosses zero, not just touches).
    Then r = sqrt(y*) and value = y*/(2*pi).
    """
    c3 = -(c0 + 3.0*c1 + 15.0*c2) / 105.0
    # Q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    a5 = c3
    a4 = 0.0
    a3 = 660.0*c3 + c2
    a2 = -60.0*c3 - 24.0*c2 + c1
    a1 = 5040.0*c3 + 72.0*c2 - 6.0*c1
    a0 = -1680.0*c3 - 32.0*c2 + 3.0*c1
    return [a5, a4, a3, a2, a1, a0]

def _q_eval(y, coeffs):
    """Evaluate Q(y) = sum coeffs[i]*y^(5-i)"""
    a5, a4, a3, a2, a1, a0 = coeffs
    return a5*y**5 + a4*y**4 + a3*y**3 + a2*y**2 + a1*y + a0

def _q_eval_vec(y_arr, coeffs):
    a5, a4, a3, a2, a1, a0 = coeffs
    return a5*y_arr**5 + a4*y_arr**4 + a3*y_arr**3 + a2*y_arr**2 + a1*y_arr + a0

def _find_largest_positive_sign_change_root(c0, c1, c2):
    """
    Find the largest real y>0 where Q(y) changes sign.
    Returns y* or None if no such root exists.
    """
    coeffs = _build_q_coeffs(c0, c1, c2)
    a5, a4, a3, a2, a1, a0 = coeffs
    
    # If a5 is essentially 0, it's a lower-degree polynomial
    # Find all real positive roots using numpy
    # Build polynomial for numpy.roots: [a5, a4, a3, a2, a1, a0]
    poly_coeffs = np.array([a5, a4, a3, a2, a1, a0], dtype=float)
    
    # Remove leading zeros
    nz = np.nonzero(np.abs(poly_coeffs) > 1e-15)[0]
    if len(nz) == 0:
        return None
    poly_coeffs = poly_coeffs[nz[0]:]
    
    if len(poly_coeffs) < 2:
        return None
    
    roots = np.roots(poly_coeffs)
    
    # Filter: real, positive
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            rr = r.real
            if rr > 1e-12:
                real_roots.append(rr)
    
    if not real_roots:
        return None
    
    # For each candidate root, check sign change
    # A root y0 is a sign-change root if Q changes sign across y0
    # For a simple root (odd multiplicity), this happens automatically
    # Check by evaluating Q at y0-eps and y0+eps
    best_y = None
    eps = 1e-8
    
    for y0 in sorted(real_roots, reverse=True):
        # Check if it's a sign-changing root
        y_left = max(y0 - eps, 1e-15)
        y_right = y0 + eps
        q_left = _q_eval(y_left, coeffs)
        q_right = _q_eval(y_right, coeffs)
        
        if q_left * q_right < 0:
            # Sign change! Use this root
            best_y = y0
            break
        else:
            # Could be a double root or numerical issue; try smaller eps
            for e in [1e-6, 1e-4, 1e-2]:
                y_left = max(y0 - e, 1e-15)
                y_right = y0 + e
                q_left = _q_eval(y_left, coeffs)
                q_right = _q_eval(y_right, coeffs)
                if q_left * q_right < 0:
                    best_y = y0
                    break
            if best_y is not None:
                break
    
    if best_y is None:
        # Fallback: if no clear sign change found, try the largest root
        # and check more carefully
        y0 = max(real_roots)
        for e in [1e-5, 1e-3, 1e-1, 1.0]:
            y_left = max(y0 - e, 1e-15)
            y_right = y0 + e
            q_left = _q_eval(y_left, coeffs)
            q_right = _q_eval(y_right, coeffs)
            if q_left * q_right < 0:
                best_y = y0
                break
    
    return best_y

def _objective(c):
    """Compute value = r^2/(2*pi) = y*/(2*pi). Lower is better.
    Return a large value if invalid."""
    c0, c1, c2 = c
    y_star = _find_largest_positive_sign_change_root(c0, c1, c2)
    if y_star is None:
        return 1e10
    # Also check: g should tend to positive limit at infinity (or be flipped)
    # The problem says: if g tends to negative limit at infinity, g is replaced by -g
    # This means we should check the leading coefficient of g (which is c3 for x^12)
    # If c3 < 0, g -> -inf, so -g is used, and -g has leading coeff -c3 > 0
    # In either case, after possible flip, the leading coeff is positive.
    # But q = g/x^2, so if we flip g, q also flips.
    # The sign-changing root of -q is the same as sign-changing root of q.
    # So the value is the same. Good.
    return y_star / (2.0 * np.pi)

def _polish(c, max_iter=200):
    """High-precision polish using Nelder-Mead with tight tolerances."""
    try:
        res = minimize(_objective, c, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.success or res.fun < _objective(c):
            return res.x
    except:
        pass
    return c

def construct():
    global _start_time
    _start_time = time.time()
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_val = _objective(seed)
    
    # Multi-start Nelder-Mead
    rng = np.random.RandomState(42)
    
    # Generate starting points around the seed and in a broader region
    starts = [seed.copy()]
    
    # Local perturbations
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(8):
            pert = rng.normal(0, scale, size=3)
            starts.append(seed + pert)
    
    # Broader exploration
    for _ in range(20):
        starts.append(rng.uniform(-3, 3, size=3))
    
    # Some structured starts
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.0, 0.1, 0.0]))
    starts.append(np.array([-0.5, -0.1, 0.01]))
    starts.append(np.array([-2.0, 0.5, -0.01]))
    
    for start in starts:
        if _deadline_exceeded():
            break
        
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except:
            continue
    
    # Polish the best result
    if not _deadline_exceeded():
        best_c = _polish(best_c, max_iter=300)
        best_val = _objective(best_c)
    
    # Final check
    final_val = _objective(best_c)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure we return valid coefficients
    if not np.all(np.isfinite(best_c)):
        best_c = seed
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
