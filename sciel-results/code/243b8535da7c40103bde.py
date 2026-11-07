"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sympy import Poly, symbols, Rational, sqrt, pi
from sympy.polys.orthopolys import hermite_poly


def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients c0, c1, c2 using exact arithmetic where possible."""
    x = symbols('x')
    
    H0 = hermite_poly(0, x)
    H4 = hermite_poly(4, x)
    H8 = hermite_poly(8, x)
    H12 = hermite_poly(12, x)
    
    # Evaluate at 0
    H0_0 = H0.subs(x, 0)
    H4_0 = H4.subs(x, 0)
    H8_0 = H8.subs(x, 0)
    H12_0 = H12.subs(x, 0)
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    # Use rational arithmetic
    c0_r = Rational(str(c0))
    c1_r = Rational(str(c1))
    c2_r = Rational(str(c2))
    
    c3_r = -(c0_r * H0_0 + c1_r * H4_0 + c2_r * H8_0) / H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = c0_r * H0 + c1_r * H4 + c2_r * H8 + c3_r * H12
    
    # Check leading coefficient
    g_poly = Poly(g, x)
    leading_coeff = g_poly.LC()
    
    if leading_coeff < 0:
        g = -g
        g_poly = Poly(g, x)
    
    # q = g / x^2
    if g_poly.degree() < 2:
        return float('inf')
    
    try:
        q_poly = g_poly / Poly(x**2, x)
    except Exception:
        return float('inf')
    
    # Find real roots of q
    try:
        roots = q_poly.nroots()
    except Exception:
        return float('inf')
    
    # Filter real roots and check sign changes
    real_roots = []
    for r in roots:
        if abs(r.as_real_imag()[1]) < 1e-10:
            real_roots.append(float(r.as_real_imag()[0]))
    
    if not real_roots:
        return float('inf')
    
    real_roots = sorted(set([round(r, 12) for r in real_roots]))
    
    # Check sign change at each root
    best_r = None
    for r in real_roots:
        eps = 1e-6
        val_left = q_poly.eval(r - eps)
        val_right = q_poly.eval(r + eps)
        if val_left * val_right < 0:
            if best_r is None or r > best_r:
                best_r = r
    
    if best_r is None:
        return float('inf')
    
    return best_r**2 / (2 * float(np.pi))


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    best_value = seed_value
    
    # Try multiple optimization strategies
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
    
    # Strategy 1: Nelder-Mead from seed
    if time.time() < deadline - 60:
        try:
            result = minimize(objective, seed, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Try random perturbations
    rng = np.random.RandomState(42)
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            perturbation = rng.normal(0, 0.05, size=3)
            x0 = seed + perturbation
            val = objective(x0)
            if val < best_value:
                best_value = val
                best_coeffs = x0.copy()
    
    # Strategy 3: Local refinement from best found
    if time.time() < deadline - 30:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
