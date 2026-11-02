"""Rewritten program: explicitly optimize the Hermite coefficients to minimize the
uncertainty bound C4 = A(f)*A(Fourier(f)) via stochastic global search (differential evolution)
using exact polynomial root-finding for the L-infinity norm."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution


def _hermite_coeffs(n):
    """Return coefficients of H_n(x) in power basis [x^n, ..., x^0]."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    
    c_prev = np.array([1.0]) # H0
    c_curr = np.array([2.0, 0.0]) # H1
    
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        term1 = np.zeros(len(c_curr) + 1)
        term1[1:] = 2.0 * c_curr
        
        term2 = np.zeros(len(c_curr) + 1)
        term2[:len(c_prev)] = 2.0 * k * c_prev
        
        c_next = term1 - term2
        c_prev = c_curr
        c_curr = c_next
        
    return c_curr

def _pad_coeffs(c, length=13):
    if len(c) == length:
        return c
    new_c = np.zeros(length)
    new_c[length - len(c):] = c
    return new_c

# Precompute coefficients for H0, H4, H8, H12
_H0_c = _pad_coeffs(_hermite_coeffs(0))
_H4_c = _pad_coeffs(_hermite_coeffs(4))
_H8_c = _pad_coeffs(_hermite_coeffs(8))
_H12_c = _pad_coeffs(_hermite_coeffs(12))

# Values at 0 (constant terms)
_H0_0 = _H0_c[-1]
_H4_0 = _H4_c[-1]
_H8_0 = _H8_c[-1]
_H12_0 = _H12_c[-1]


def _objective(coeffs):
    """
    coeffs: [c0, c1, c2]
    c3 is derived from g(0)=0.
    Returns (max |g(x)|)^2
    """
    c0, c1, c2 = coeffs
    
    # Calculate c3 such that P(0) = 0
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Construct P(x) coefficients
    P_c = c0 * _H0_c + c1 * _H4_c + c2 * _H8_c + c3 * _H12_c
    
    # Calculate Q(x) = P'(x) - x P(x)
    # P_c length 13 (degree 12)
    # Q(x) degree 13
    
    # x P(x) coeffs
    xP_c = np.zeros(14)
    xP_c[:13] = P_c
    
    # P'(x) coeffs
    P_d_raw = np.polyder(P_c) # length 12
    P_d_c = np.zeros(14)
    # P_d_raw[0] is coeff of x^11. In P_d_c, x^11 is index 2 (x^13, x^12, x^11...)
    P_d_c[2:] = P_d_raw
    
    Q_c = P_d_c - xP_c
    
    try:
        roots = np.roots(Q_c)
    except Exception:
        return 1e12
        
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    max_val = 0.0
    for r in real_roots:
        p_val = np.polyval(P_c, r)
        g_val = p_val * np.exp(-0.5 * r * r)
        max_val = max(max_val, abs(g_val))
        
    return max_val * max_val


def construct():
    # Bounds for c0, c1, c2
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    try:
        res = differential_evolution(_objective, bounds, maxiter=100, tol=1e-12, seed=42, polish=True)
        if res.success:
            return np.array(res.x)
    except Exception:
        pass
        
    # Fallback to seed
    return np.array([-1.13, 0.04, 0.000308])
# EVOLVE-BLOCK-END
