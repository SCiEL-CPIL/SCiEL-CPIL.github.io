"""Rewritten program: Gram-Schmidt parameterization + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial
import sympy as sp
from sympy.polys.orthopolys import hermite_poly
import warnings
warnings.filterwarnings("ignore")


# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = 2^n * sum_{k=0}^{n//2} (-1)^k * n! / (k! * (n-2k)!) * x^(n-2k)
# All even Hermite polynomials have only even powers

def _hermite_coeffs(n):
    """Return coefficients of H_n from highest degree to constant."""
    coeffs = np.zeros(n + 1)
    for k in range(n // 2 + 1):
        power = n - 2 * k
        coeff = (2 ** n) * ((-1) ** k) * factorial(n, exact=False) / (factorial(k, exact=False) * factorial(n - 2 * k, exact=False))
        coeffs[n - power] = coeff  # index from highest degree
    return coeffs

# Precompute
H0_c = _hermite_coeffs(0)   # [1]
H4_c = _hermite_coeffs(4)   # [16, 0, -48, 0, 12]
H8_c = _hermite_coeffs(8)   # [256, 0, -3072, 0, 13440, 0, -26880, 0, 1680]
H12_c = _hermite_coeffs(12) # [4096, 0, -98304, 0, 1032192, 0, -5806080, 0, 16515072, 0, -25401600, 0, 13305600]

# Values at x=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 13305600.0


def _compute_value(c0, c1, c2, use_sympy=False):
    """
    Compute the objective value for given coefficients (c0, c1, c2).
    Lower is better. Returns a large penalty for invalid configurations.
    """
    try:
        if use_sympy:
            # Use exact rational arithmetic via sympy
            c0_s = sp.Rational(c0).limit_denominator(10**12)
            c1_s = sp.Rational(c1).limit_denominator(10**12)
            c2_s = sp.Rational(c2).limit_denominator(10**12)
            
            H0 = hermite_poly(0)
            H4 = hermite_poly(4)
            H8 = hermite_poly(8)
            H12 = hermite_poly(12)
            
            # Compute c3 so that g(0) = 0
            h0_0 = H0.subs(sp.Symbol('x'), 0)
            h4_0 = H4.subs(sp.Symbol('x'), 0)
            h8_0 = H8.subs(sp.Symbol('x'), 0)
            h12_0 = H12.subs(sp.Symbol('x'), 0)
            
            c3_s = -(c0_s * h0_0 + c1_s * h4_0 + c2_s * h8_0) / h12_0
            
            x = sp.Symbol('x')
            g = c0_s * H0 + c1_s * H4 + c2_s * H8 + c3_s * H12
            
            # Check if g tends to negative limit at infinity
            if c3_s < 0:
                g = -g
                c0_s = -c0_s
                c1_s = -c1_s
                c2_s = -c2_s
                c3_s = -c3_s
            
            # q = g / x^2
            gp = sp.diff(g, x)
            gp_0 = gp.subs(x, 0)
            
            if gp_0 != 0:
                return 1e10
            
            q = sp.simplify(g / x**2)
            
            # Find real roots of q where q changes sign.
            roots = sp.nroots(q, n=30)
            valid_roots = []
            for root in roots:
                if abs(root.as_real_imag()[1]) < 1e-10:
                    r_real = float(root.as_real_imag()[0])
                    eps = 1e-6
                    q_minus = float(q.subs(x, r_real - eps))
                    q_plus = float(q.subs(x, r_real + eps))
                    if q_minus * q_plus < 0:
                        valid_roots.append(r_real)
            
            if not valid_roots:
                return 1e10
            
            r = max(valid_roots)
            value = r**2 / (2 * np.pi)
            return value
        else:
            # Numerical approach
            c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
            
            # Build g coefficients (degree 12 down to 0)
            g_coeffs = np.zeros(13)
            
            # H0: degree 0
            g_coeffs[12] += c0 * H0_c[0]
            
            # H4: degree 4, coeffs from x^4 to x^0
            for i, c in enumerate(H4_c):
                g_coeffs[8 + i] += c1 * c
            
            # H8: degree 8
            for i, c in enumerate(H8_c):
                g_coeffs[4 + i] += c2 * c
            
            # H12: degree 12
            for i, c in enumerate(H12_c):
                g_coeffs[i] += c3 * c
            
            # Check leading coefficient
            if g_coeffs[0] < 0:
                g_coeffs = -g_coeffs
                c0 = -c0
                c1 = -c1
                c2 = -c2
                c3 = -c3
            
            # q = g / x^2 requires g[12] = 0 and g[11] = 0
            if abs(g_coeffs[12]) > 1e-10 or abs(g_coeffs[11]) > 1e-10:
                return 1e10
            
            # q is degree 10: coefficients g_coeffs[0:11]
            q_coeffs = g_coeffs[0:11]
            
            # Find real roots of q
            roots = np.roots(q_coeffs)
            valid_roots = []
            for root in roots:
                if abs(root.imag) < 1e-10:
                    r_real = root.real
                    eps = 1e-6
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    if q_minus * q_plus < 0:
                        valid_roots.append(r_real)
            
            if not valid_roots:
                return 1e10
            
            r = max(valid_roots)
            value = r**2 / (2 * np.pi)
            return value
            
    except Exception as e:
        return 1e10


def _objective(params):
    """
    Objective function for differential evolution.
    params = [c0, c1, c2]
    """
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2, use_sympy=False)


def _local_refine(c0, c1, c2, max_iter=50):
    """Local refinement using Nelder-Mead."""
    def obj(params):
        return _compute_value(params[0], params[1], params[2], use_sympy=False)
    
    try:
        result = minimize(obj, np.array([c0, c1, c2]), method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
        return result.x, result.fun
    except Exception:
        return np.array([c0, c1, c2]), _compute_value(c0, c1, c2, use_sympy=False)


def construct():
    """
    Construct three coefficients [c0, c1, c2] using differential evolution with local refinement.
    """
    # Initial seed
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2], use_sympy=False)
    
    best_value = seed_value
    best_params = seed.copy()
    
    # Multiple DE runs with different seeds and bounds
    bounds_sets = [
        [(-5, 5), (-5, 5), (-5, 5)],
        [(-2, 2), (-2, 2), (-2, 2)],
        [(-1, 1), (-1, 1), (-1, 1)],
        [(-10, 10), (-10, 10), (-10, 10)],
    ]
    
    de_seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for bounds in bounds_sets:
        for de_seed in de_seeds:
            try:
                result = differential_evolution(
                    _objective,
                    bounds,
                    seed=de_seed,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    polish=False,
                    init='sobol'
                )
                
                if result.fun < best_value:
                    # Local refine
                    refined_params, refined_val = _local_refine(
                        result.x[0], result.x[1], result.x[2], max_iter=100
                    )
                    if refined_val < best_value:
                        best_value = refined_val
                        best_params = refined_params
                elif result.fun < best_value + 0.1:
                    # Also try refining near the DE result
                    refined_params, refined_val = _local_refine(
                        result.x[0], result.x[1], result.x[2], max_iter=100
                    )
                    if refined_val < best_value:
                        best_value = refined_val
                        best_params = refined_params
                        
            except Exception:
                pass
    
    # Also try random sampling with local refinement
    np.random.seed(42)
    for i in range(50):
        c0 = np.random.uniform(-3, 3)
        c1 = np.random.uniform(-3, 3)
        c2 = np.random.uniform(-3, 3)
        val = _compute_value(c0, c1, c2, use_sympy=False)
        if val < best_value + 0.5:
            refined_params, refined_val = _local_refine(c0, c1, c2, max_iter=100)
            if refined_val < best_value:
                best_value = refined_val
                best_params = refined_params
    
    # Verify best with sympy
    if best_value < seed_value:
        sympy_value = _compute_value(best_params[0], best_params[1], best_params[2], use_sympy=True)
        if sympy_value < seed_value:
            return np.array(best_params)
    
    return seed
# EVOLVE-BLOCK-END
