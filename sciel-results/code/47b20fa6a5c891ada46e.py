"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial2
import time

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k * (2x)^(n-2k) / (k! * (n-2k)!)
# Actually physicists' Hermite: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# Coefficients: H_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * n! / (k! (n-2k)!) * (2x)^(n-2k)

def _hermite_coeffs(n):
    """Return list of coefficients for H_n(x), index i = power of x^i."""
    coeffs = [0.0] * (n + 1)
    for k in range(n // 2 + 1):
        coeff = (-1) ** k * float(factorial2(n, 2)) / (float(factorial2(2*k, 2)) * 2**k * float(factorial2(n - 2*k, 2)))
        # Wait, let me use the standard formula:
        # H_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * n! / (k! * (n-2k)!) * (2x)^(n-2k)
        pass
    # Use sympy-like approach but with floats
    from math import factorial
    coeffs = [0.0] * (n + 1)
    for k in range(n // 2 + 1):
        c = (-1.0) ** k * factorial(n) / (factorial(k) * factorial(n - 2 * k)) * (2.0 ** (n - 2 * k))
        coeffs[n - 2 * k] = c
    return coeffs

_H0 = _hermite_coeffs(0)   # [1]
_H4 = _hermite_coeffs(4)   # 16x^4 - 48x^2 + 12
_H8 = _hermite_coeffs(8)   # 256x^8 - 4608x^6 + 23040x^4 - 32256x^2 + 40320
_H12 = _hermite_coeffs(12)

# Values at x=0
_H0_0 = _H0[0]  # 1
_H4_0 = _H4[0]  # 12
_H8_0 = _H8[0]  # 40320
_H12_0 = _H12[0]  # 47900160

def _build_q_coefficients(c0, c1, c2):
    """
    Build the coefficients of q(x) = g(x) / x^2, where
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0.
    
    q(x) = g(x)/x^2 is a polynomial of degree 10 (since g has degree 12 and g(0)=0, g(0')=0 too? 
    Actually g(0)=0 means the constant term is 0, so g(x) = x^2 * q(x) where q is degree 10.
    """
    # g(x) coefficients: degree 12 down to 0
    # H0: degree 0, H4: degree 4, H8: degree 8, H12: degree 12
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g coefficients array (index = power of x)
    g_coeffs = [0.0] * 13  # degree 0 to 12
    for i in range(len(_H0)):
        g_coeffs[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g_coeffs[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g_coeffs[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    # Check if g tends to negative limit at infinity: leading coefficient is c3 * _H12[12]
    # If leading coeff < 0, flip sign of all coefficients
    if g_coeffs[12] < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    # q(x) = g(x) / x^2, so q_coeffs[i] = g_coeffs[i+2] for i = 0..10
    q_coeffs = g_coeffs[2:13]  # degree 0 to 10
    
    return q_coeffs

def _eval_poly(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result

def _find_largest_sign_change_root(q_coeffs):
    """
    Find the largest real root of q where q changes sign.
    q_coeffs: list of coefficients from degree 0 to degree n.
    """
    from numpy.polynomial import polynomial as P
    
    # Find all real roots
    # Use numpy roots which works with coefficients in descending order
    # q_coeffs is in ascending order (index = power)
    q_desc = q_coeffs[::-1]
    
    # Remove leading zeros
    while len(q_desc) > 1 and abs(q_desc[0]) < 1e-15:
        q_desc = q_desc[1:]
    
    if len(q_desc) <= 1:
        return None
    
    roots = np.roots(q_desc)
    
    # Filter real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
    
    if not real_roots:
        return None
    
    # For each real root, check if q changes sign
    # A sign change at root r means q(r-eps) and q(r+eps) have opposite signs
    sign_change_roots = []
    for r in real_roots:
        eps = 1e-8
        q_left = _eval_poly(q_coeffs, r - eps)
        q_right = _eval_poly(q_coeffs, r + eps)
        if q_left * q_right < 0:
            sign_change_roots.append(r)
    
    if not sign_change_roots:
        return None
    
    return max(sign_change_roots)

def _compute_value(c0, c1, c2):
    """Compute the evaluation value r^2 / (2*pi)."""
    q_coeffs = _build_q_coefficients(c0, c1, c2)
    r = _find_largest_sign_change_root(q_coeffs)
    if r is None:
        return float('inf')
    return r * r / (2.0 * np.pi)

def _objective(coeffs):
    """Objective function for optimization."""
    c0, c1, c2 = coeffs
    try:
        val = _compute_value(c0, c1, c2)
        if val == float('inf') or np.isnan(val):
            return 1e10
        return val
    except Exception:
        return 1e10

# EVOLVE-BLOCK-END

def construct():
    # Seed point
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    start_time = time.time()
    deadline = start_time + 180.0  # Leave margin before 210s kill
    
    best_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    # Try multiple starting points and optimization methods
    starting_points = [
        seed,
        seed + np.array([0.1, 0.01, 0.001]),
        seed - np.array([0.1, 0.01, 0.001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-2.0, 0.2, 0.002]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([-0.8, 0.08, 0.0008]),
    ]
    
    for sp in starting_points:
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = minimize(_objective, sp, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Refine with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
