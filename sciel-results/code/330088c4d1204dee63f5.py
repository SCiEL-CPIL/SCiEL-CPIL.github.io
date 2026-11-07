"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sympy import Poly, symbols, Rational, sqrt, pi
from sympy.polys.orthopolys import hermite_poly


# Precompute Hermite polynomials and their values at 0
_x = symbols('x')
_H0 = hermite_poly(0, _x)
_H4 = hermite_poly(4, _x)
_H8 = hermite_poly(8, _x)
_H12 = hermite_poly(12, _x)
_H0_0 = _H0.subs(_x, 0)
_H4_0 = _H4.subs(_x, 0)
_H8_0 = _H8.subs(_x, 0)
_H12_0 = _H12.subs(_x, 0)
_x2 = Poly(_x**2, _x)

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients c0, c1, c2 using exact arithmetic where possible."""
    try:
        c0_r = Rational(str(c0))
        c1_r = Rational(str(c1))
        c2_r = Rational(str(c2))
        
        c3_r = -(c0_r * _H0_0 + c1_r * _H4_0 + c2_r * _H8_0) / _H12_0
        
        g = c0_r * _H0 + c1_r * _H4 + c2_r * _H8 + c3_r * _H12
        g_poly = Poly(g, _x)
        
        if g_poly.LC() < 0:
            g_poly = -g_poly
        
        if g_poly.degree() < 2:
            return float('inf')
        
        try:
            q_poly = g_poly / _x2
        except Exception:
            return float('inf')
        
        try:
            roots = q_poly.nroots()
        except Exception:
            return float('inf')
        
        real_roots = []
        for r in roots:
            re, im = r.as_real_imag()
            if abs(im) < 1e-10:
                real_roots.append(float(re))
        
        if not real_roots:
            return float('inf')
        
        real_roots = sorted(set([round(r, 12) for r in real_roots]))
        
        best_r = None
        for r in real_roots:
            eps = 1e-8
            val_left = q_poly.eval(r - eps)
            val_right = q_poly.eval(r + eps)
            if val_left * val_right < 0:
                if best_r is None or r > best_r:
                    best_r = r
        
        if best_r is None:
            return float('inf')
        
        return best_r**2 / (2 * float(np.pi))
    except Exception:
        return float('inf')


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 195  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    best_value = seed_value
    
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
    
    # Strategy 1: Nelder-Mead from seed with more iterations
    if time.time() < deadline - 60:
        try:
            result = minimize(objective, seed, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    if time.time() < deadline - 30:
        for i in range(50):
            if time.time() > deadline - 30:
                break
            perturbation = rng.normal(0, 0.1, size=3)
            x0 = seed + perturbation
            val = objective(x0)
            if val < best_value:
                best_value = val
                best_coeffs = x0.copy()
            # Local refinement from this point
            if time.time() < deadline - 30:
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    if result.fun < best_value:
                        best_value = result.fun
                        best_coeffs = result.x.copy()
                except Exception:
                    pass
    
    # Strategy 3: Final high-precision polish from best found
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
