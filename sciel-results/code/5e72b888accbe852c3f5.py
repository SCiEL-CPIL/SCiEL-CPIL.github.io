"""Combined program: Structural change to minimize the largest positive simple root of the reduced polynomial P(z)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from sympy import symbols, Poly, hermite_poly
import sys

# Precompute Hermite polynomials coefficients (low to high)
def _get_hermite_coeffs_low_to_high(n):
    x = symbols('x')
    p_expr = hermite_poly(n, x)
    p_poly = Poly(p_expr, x)
    # all_coeffs returns high to low
    coeffs_high_to_low = p_poly.all_coeffs()
    return np.array([float(c) for c in coeffs_high_to_low[::-1]])

_H0 = _get_hermite_coeffs_low_to_high(0)   # [1]
_H4 = _get_hermite_coeffs_low_to_high(4)   # [12, 0, -48, 0, 16]
_H8 = _get_hermite_coeffs_low_to_high(8)
_H12 = _get_hermite_coeffs_low_to_high(12)

# H12(0) is the constant term
_H12_0 = _H12[0]
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]

# Degree of P(z) is 5. z = x^2.
# g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
# g(x) is even. g(x) = sum_{k=0}^6 g_{2k} x^{2k}.
# P(z) = sum_{k=0}^6 g_{2k} z^k.
# Coeffs of P(z) are [g_2, g_4, ..., g_12].

def _get_P_coeffs(c0, c1, c2):
    # c3 = -(c0 H0(0) + c1 H4(0) + c2 H8(0)) / H12(0)
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Construct g coefficients (low to high)
    # Max degree 12
    g = np.zeros(13)
    
    def add_poly(coeffs, scale):
        d = len(coeffs)
        for i in range(d):
            g[i] += scale * coeffs[i]
            
    add_poly(_H0, c0)
    add_poly(_H4, c1)
    add_poly(_H8, c2)
    add_poly(_H12, c3)
    
    # P(z) coefficients are g[2], g[4], ..., g[12]
    # g[0] and g[1] should be 0 (g(0)=0 and g is even)
    P_coeffs = g[2:13:2]
    return P_coeffs

def _largest_positive_simple_root(P_coeffs):
    """
    Returns the largest positive real simple root of the polynomial P_coeffs (low to high).
    Returns 0.0 if there are no such roots.
    """
    # Remove leading zeros
    while len(P_coeffs) > 1 and abs(P_coeffs[-1]) < 1e-15:
        P_coeffs = P_coeffs[:-1]
        
    if len(P_coeffs) < 2:
        return 0.0

    # np.roots expects high to low
    roots = np.roots(P_coeffs[::-1])
    
    max_root = 0.0
    for r in roots:
        if abs(r.imag) > 1e-9 * max(1.0, abs(r.real)):
            continue
        zr = float(r.real)
        if zr <= 1e-12:
            continue
        
        # Check if it's a simple root (sign change)
        # Evaluate derivative at zr
        # Derivative coeffs (low to high)
        d_coeffs = np.polyder(P_coeffs[::-1])[::-1] # high to low -> derivative -> low to high?
        # np.polyder takes high to low, returns high to low.
        # Let's just evaluate P and P'
        
        # Use np.polyval with high-to-low coeffs
        P_high = P_coeffs[::-1]
        d_high = np.polyder(P_high)
        
        P_val = np.polyval(P_high, zr)
        d_val = np.polyval(d_high, zr)
        
        # Simple root if P_val ~ 0 and d_val != 0
        # Tolerance for P_val should be relative to magnitude
        if abs(P_val) > 1e-7 * max(1.0, np.max(np.abs(P_high)) * zr**5):
            continue
            
        if abs(d_val) < 1e-9:
            continue
            
        if zr > max_root:
            max_root = zr
            
    return max_root

def _objective(c):
    try:
        c0, c1, c2 = c
        P_coeffs = _get_P_coeffs(c0, c1, c2)
        r = _largest_positive_simple_root(P_coeffs)
        # We want to minimize r
        # If r is 0, it's the best possible
        return r
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = seed.copy()
    best_val = _objective(best_x)
    
    # Strategy:
    # 1. Differential Evolution to find a global minimum of the largest root.
    #    Bounds: We need to explore a reasonable range.
    #    The seed is around [-1, 0.04, 0.0003].
    #    Let's try bounds [-10, 10], [-1, 1], [-0.01, 0.01] or similar.
    #    Actually, the scale of c2 is small in seed.
    
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    # Try DE first
    if time.time() < deadline - 60:
        try:
            # Differential Evolution
            # maxiter might be high, but we check time
            res_de = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass
            
    # 2. Multi-start Nelder-Mead for local refinement
    # Generate diverse starts
    np.random.seed(42)
    starts = [best_x.copy()]
    
    # Random starts in a box
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        p = np.random.uniform(-5, 5, size=3)
        p[2] *= 0.05 # c2 is smaller scale
        starts.append(p)
        
    # Perturbations of seed
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            p = seed + scale * np.random.randn(3)
            p[2] *= 0.05
            starts.append(p)
            
    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    # 3. High precision polish
    if best_val < 1e10 and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
