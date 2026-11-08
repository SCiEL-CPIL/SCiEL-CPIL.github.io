"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite, erf

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 175.0  # seconds, with safety margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # H_k(0) for even k: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-103950
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1680.0
    h12_0 = -103950.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12

# Precompute coefficients for Hermite polynomials H_0, H_4, H_8, H_12
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H_12(x) = 4096x^12 - 76800x^10 + 518400x^8 - 1548480x^6 + 2342400x^4 - 1548480x^2 + 103950

# Coefficients for x^12 down to x^0
# Index 0: x^12, Index 1: x^10, ..., Index 6: x^0
# H_0: [0, 0, 0, 0, 0, 0, 1]
# H_4: [0, 0, 0, 16, -48, 0, 12]
# H_8: [0, 256, 0, -3072, 0, 11520, -13440, 1680] -> Wait, degree 8.
# Let's align to degree 12.
# H_0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (13 coeffs for x^12..x^0)
# H_4: [0, 0, 0, 0, 0, 0, 0, 0, 16, -48, 0, 12, 0] -> No, x^4, x^2, x^0.
# Indices: 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
# H_0: [0,0,0,0,0,0,0,0,0,0,0,0,1]
# H_4: [0,0,0,0,0,0,0,0,16,0,-48,0,12]
# H_8: [0,0,0,0,256,0,-3072,0,11520,0,-13440,0,1680]
# H_12:[4096,0,-76800,0,518400,0,-1548480,0,2342400,0,-1548480,0,103950]

_H0_coeffs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1.0])
_H4_coeffs = np.array([0,0,0,0,0,0,0,0,16,0,-48,0,12.0])
_H8_coeffs = np.array([0,0,0,0,256,0,-3072,0,11520,0,-13440,0,1680.0])
_H12_coeffs = np.array([4096,0,-76800,0,518400,0,-1548480,0,2342400,0,-1548480,0,103950.0])

def _get_poly_coeffs(c0, c4, c8, c12):
    """Return coefficients of P(x) for x^12 down to x^0."""
    return c0 * _H0_coeffs + c4 * _H4_coeffs + c8 * _H8_coeffs + c12 * _H12_coeffs

def _int_xn_exp_neg_half_x2(x, n):
    """
    Compute the definite integral from 0 to x of t^n exp(-t^2/2) dt.
    Uses recurrence: I_n(x) = -x^{n-1} exp(-x^2/2) + (n-1) I_{n-2}(x)
    I_0(x) = sqrt(pi/2) erf(x/sqrt(2))
    I_1(x) = -exp(-x^2/2) + 1  (since integral from 0 to x is 1 - exp(-x^2/2))
    Wait, I_1(x) = int_0^x t exp(-t^2/2) dt = [-exp(-t^2/2)]_0^x = 1 - exp(-x^2/2).
    """
    if n == 0:
        return np.sqrt(np.pi / 2.0) * erf(x / np.sqrt(2.0))
    elif n == 1:
        return 1.0 - np.exp(-x * x / 2.0)
    else:
        # Recurrence
        # I_n(x) = -x^{n-1} exp(-x^2/2) + (n-1) I_{n-2}(x)
        return -x ** (n - 1) * np.exp(-x * x / 2.0) + (n - 1) * _int_xn_exp_neg_half_x2(x, n - 2)

def _antiderivative_P(x, coeffs):
    """
    Compute the antiderivative of P(x) exp(-x^2/2) from 0 to x.
    coeffs are for x^12, x^11, ..., x^0.
    """
    integral = 0.0
    for i, c in enumerate(coeffs):
        if c != 0:
            n = 12 - i
            integral += c * _int_xn_exp_neg_half_x2(x, n)
    return integral

def _compute_bound_fast(c0, c4, c8):
    """Compute the upper bound using analytic integration."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    coeffs = _get_poly_coeffs(c0, c4, c8, c12)
    
    # Find real roots of P(x)
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    real_roots.sort()
    
    # Boundaries
    bounds = np.concatenate(([-np.inf], real_roots, [np.inf]))
    
    total_integral = 0.0
    # Evaluate antiderivative at boundaries
    # Note: antiderivative is from 0 to x.
    # We need integral from a to b.
    # F(b) - F(a).
    # However, F(-inf) and F(inf) are tricky with the "from 0" definition.
    # Let's define G(x) = int_{-inf}^x P(t) exp(-t^2/2) dt.
    # Then integral from a to b is G(b) - G(a).
    # G(x) = int_{-inf}^0 ... + int_0^x ...
    # int_{-inf}^0 P(t) exp(-t^2/2) dt is a constant C.
    # Let's just compute F(x) = int_0^x and handle the signs carefully.
    # Actually, simpler:
    # Split integral into segments [b_i, b_{i+1}].
    # On each segment, sign of P is constant.
    # Determine sign at midpoint.
    
    # To avoid issues with infinity, we can note that
    # int_{-inf}^inf |P(x)| exp(-x^2/2) dx = sum |int_{a_i}^{a_{i+1}} P(x) exp(-x^2/2) dx|
    
    # Let's compute F(x) = int_0^x P(t) exp(-t^2/2) dt.
    # Then int_a^b P(t) exp(-t^2/2) dt = F(b) - F(a).
    # We need to handle -inf and inf.
    # F(inf) = int_0^inf P(t) exp(-t^2/2) dt.
    # F(-inf) = int_0^{-inf} P(t) exp(-t^2/2) dt = - int_{-inf}^0 P(t) exp(-t^2/2) dt.
    
    # Let's compute F_vals for all finite roots and 0.
    # And compute the total integral from -inf to inf of P(x) exp(-x^2/2) dx? No, that's 0 if P is odd? No.
    # Actually, we can just sum the absolute values of the integrals over each interval.
    
    # Let's refine the bounds list to include 0 if it's not a root, to split the domain at 0?
    # Not strictly necessary if we use F(b)-F(a).
    # But F(inf) and F(-inf) need to be calculated.
    
    # F(inf) = sum_{even n} c_n * int_0^inf x^n exp(-x^2/2) dx
    # int_0^inf x^n exp(-x^2/2) dx = 1/2 * 2^{n/2} * Gamma((n+1)/2)
    # For odd n, it's not zero, it's a value.
    # Wait, P(x) is not necessarily even or odd.
    
    # Let's just calculate F at all finite roots and 0.
    # And calculate F_inf = int_0^inf P(t) exp(-t^2/2) dt.
    # And F_neg_inf = int_0^{-inf} P(t) exp(-t^2/2) dt = - int_{-inf}^0 ...
    
    # Actually, it's safer to just integrate over finite intervals and handle the tails separately?
    # No, the tails are part of the intervals [-inf, r_1] and [r_m, inf].
    
    # Let's compute F(x) for all x in bounds, treating inf specially.
    
    F_inf = 0.0
    for i, c in enumerate(coeffs):
        if c != 0:
            n = 12 - i
            # int_0^inf x^n exp(-x^2/2) dx
            # u = x^2/2, x = sqrt(2u), dx = 1/sqrt(2u) du
            # int_0^inf (2u)^{n/2} exp(-u) 1/sqrt(2u) du
            # = 2^{n/2 - 1/2} int_0^inf u^{(n-1)/2} exp(-u) du
            # = 2^{(n-1)/2} Gamma((n+1)/2)
            F_inf += c * (2.0 ** ((n - 1) / 2.0)) * scipy.special.gamma((n + 1) / 2.0)
            
    # F_neg_inf = - F_inf? No.
    # int_{-inf}^0 x^n exp(-x^2/2) dx.
    # If n is even, it's same as int_0^inf.
    # If n is odd, it's - int_0^inf.
    F_neg_inf = 0.0
    for i, c in enumerate(coeffs):
        if c != 0:
            n = 12 - i
            val = (2.0 ** ((n - 1) / 2.0)) * scipy.special.gamma((n + 1) / 2.0)
            if n % 2 == 1:
                val = -val
            F_neg_inf += c * val
            
    # Now we have F(-inf) = F_neg_inf, F(inf) = F_inf.
    # F(0) = 0.
    
    # List of points: [-inf, r_1, ..., r_m, inf]
    # We need to evaluate F at each point.
    
    points = bounds
    F_vals = np.zeros(len(points))
    
    for i, p in enumerate(points):
        if np.isinf(p):
            if p > 0:
                F_vals[i] = F_inf
            else:
                F_vals[i] = F_neg_inf
        else:
            F_vals[i] = _antiderivative_P(p, coeffs)
            
    # Now sum |F_vals[i+1] - F_vals[i]|
    total_integral = np.sum(np.abs(np.diff(F_vals)))
    
    A_f = total_integral
    A_F = np.sqrt(2 * np.pi) * A_f
    return A_f * A_F

import scipy.special

def _compute_bound_analytic(c0, c4, c8):
    """Compute the upper bound - uses analytic method."""
    return _compute_bound_fast(c0, c4, c8)

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_fast,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try multi-start with different initial points
    if _time_left() > 50:
        try:
            # Try several random starting points
            np.random.seed(42)
            for _ in range(20): # Increased starts due to speedup
                if _time_left() < 20:
                    break
                start = np.random.uniform([-5, 5, -0.5, 0.5], [5, 5, 0.5, -0.5])[:3]
                res = minimize(
                    _compute_bound_fast,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 80:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            res_de = differential_evolution(
                _compute_bound_fast,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-8,
                polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if _time_left() > 15 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
