"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time
import sys

def _build_g_coeffs(c0, c1, c2):
    """
    Build the coefficient array for g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12.
    c3 is chosen so g(0) = 0.
    Returns list of coefficients [a0, a1, ..., a12] for sum ai*x^i.
    """
    # Hermite polynomials (physicists'): H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1536x^6 + 3360x^4 - 1920x^2 + 168
    # H12 = 4096x^12 - 393216x^10 + 1848384x^8 - 3317760x^6 + 2612736x^4 - 798336x^2 + 399168
    
    # Coefficients for each Hermite poly: [const, x^2, x^4, x^6, x^8, x^10, x^12]
    H0 = [1, 0, 0, 0, 0, 0, 0]
    H4 = [12, -48, 16, 0, 0, 0, 0]
    H8 = [168, -1920, 3360, -1536, 256, 0, 0]
    H12 = [399168, -798336, 2612736, -3317760, 1848384, -393216, 4096]
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(0) = c0*H0[0] + c1*H4[0] + c2*H8[0] + c3*H12[0] = 0
    # c3 = -(c0*1 + c1*12 + c2*168) / 399168
    
    c3 = -(c0 * H0[0] + c1 * H4[0] + c2 * H8[0]) / H12[0]
    
    # Build g coefficients: g[i] corresponds to x^(2*i) since all are even
    g_coeffs = np.zeros(7)  # indices 0..6 for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    for i in range(7):
        g_coeffs[i] = c0 * H0[i] + c1 * H4[i] + c2 * H8[i] + c3 * H12[i]
    
    # g is even, so g(x) = sum_{i=0}^{6} g_coeffs[i] * x^(2*i)
    # q(x) = g(x) / x^2 = sum_{i=0}^{6} g_coeffs[i] * x^(2*(i-1)) for i>=1
    # q(x) = g_coeffs[1] + g_coeffs[2]*x^2 + g_coeffs[3]*x^4 + g_coeffs[4]*x^6 + g_coeffs[5]*x^8 + g_coeffs[6]*x^10
    
    # Check that g(0) = g_coeffs[0] should be ~0
    # q is a polynomial in x^2, degree 10 in x, or degree 5 in y=x^2
    
    # Return q as polynomial in y = x^2: q(y) = g_coeffs[1] + g_coeffs[2]*y + ... + g_coeffs[6]*y^5
    q_coeffs_y = g_coeffs[1:7]  # [q0, q1, q2, q3, q4, q5] for q(y) = q0 + q1*y + ... + q5*y^5
    
    return q_coeffs_y, g_coeffs

def _largest_sign_changing_root(q_coeffs_y):
    """
    Find the largest real root of q(x) = g(x)/x^2 where q changes sign.
    q as function of x: q(x) = sum_{j=0}^{5} q_coeffs_y[j] * x^(2*j)
    Let y = x^2, then q(x) = Q(y) where Q(y) = sum q_coeffs_y[j] * y^j
    Roots of q(x) are x = ±sqrt(y) for each positive root y of Q(y).
    We need the largest |x| where q changes sign.
    
    Since q(x) = Q(x^2), q changes sign at x = r iff Q changes sign at y = r^2.
    The largest sign-changing root r corresponds to the largest positive root y* of Q(y)
    at which Q changes sign, and r = sqrt(y*).
    """
    # Find roots of Q(y) = q_coeffs_y[0] + q_coeffs_y[1]*y + ... + q_coeffs_y[5]*y^5
    # Use numpy to find roots
    try:
        roots_y = np.roots(q_coeffs_y[::-1])  # np.roots expects highest degree first
    except:
        return None
    
    best_r = 0.0
    for y in roots_y:
        if np.isreal(y):
            y_real = float(np.real(y))
            if y_real > 1e-15:  # positive root
                # Check sign change: evaluate Q at y_real - eps and y_real + eps
                # Or check if it's a simple root (multiplicity 1)
                # A root changes sign if it has odd multiplicity
                # For now, assume simple roots change sign
                r = np.sqrt(y_real)
                if r > best_r:
                    best_r = r
    
    return best_r

def _compute_value(c0, c1, c2):
    """Compute the value r^2 / (2*pi) for given coefficients."""
    q_coeffs_y, g_coeffs = _build_g_coeffs(c0, c1, c2)
    
    # Check if g tends to negative limit at infinity
    # Leading coefficient of g is g_coeffs[6] (for x^12)
    # If leading coeff < 0, g -> -inf as x -> inf, so we replace g by -g
    if g_coeffs[6] < 0:
        q_coeffs_y = -q_coeffs_y
        g_coeffs = -g_coeffs
    
    r = _largest_sign_changing_root(q_coeffs_y)
    if r is None or r <= 0:
        return None
    
    return r**2 / (2 * np.pi)

def _objective(params):
    """Objective function to minimize: the value r^2/(2*pi)."""
    c0, c1, c2 = params
    val = _compute_value(c0, c1, c2)
    if val is None:
        return 1e10
    return val

def construct():
    """Find optimal coefficients [c0, c1, c2]."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Compute seed value
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    if seed_value is None:
        seed_value = 1e10
    best_params = seed.copy()
    best_value = seed_value
    
    # Local optimization from seed
    try:
        result = minimize(_objective, seed, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
        if result.success or result.fun < best_value:
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_params
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization near seed or in a reasonable range
        if i < 5:
            # Perturb seed
            perturbation = np.random.normal(0, 0.01, size=3)
            x0 = seed + perturbation
        else:
            # Random in a box
            x0 = np.random.uniform([-2, -2, -0.1], [2, 2, 0.1], size=3)
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                      'adaptive': True})
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass
    
    # Final polish
    try:
        result = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                  'adaptive': True})
        if result.fun < best_value:
            best_params = result.x
            best_value = result.fun
    except Exception:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
