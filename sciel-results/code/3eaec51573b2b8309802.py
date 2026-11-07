"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sympy import symbols, Poly, hermite_poly, Rational, pi as sp_pi
from sympy.polys.orthopolys import hermite_poly as _hp

# Precompute Hermite polynomials as coefficient lists (descending powers)
# H_n(x) = sum a_k x^k
def _hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial H_n as list [a_n, a_{n-1}, ..., a_0]."""
    # Use sympy to get exact coefficients
    x = symbols('x')
    p_expr = _hp(n, x)
    p_poly = Poly(p_expr, x)
    c_list = p_poly.all_coeffs()
    return [float(c) for c in c_list]

# Precompute
_H0 = _hermite_coeffs(0)   # [1]
_H4 = _hermite_coeffs(4)   # [16, 0, -48, 0, 12]
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

def _compute_value(c0, c1, c2):
    """Compute the objective value r^2/(2*pi) for given c0,c1,c2."""
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    H0_0 = _H0[-1]  # 1
    H4_0 = _H4[-1]  # 12
    H8_0 = _H8[-1]  # 168
    H12_0 = _H12[-1]  # 21257664
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Build coefficient array for g (descending powers of x)
    # Max degree is 12
    deg = 12
    g_coeffs = np.zeros(deg + 1)
    
    def add_poly(coeffs, scale):
        d = len(coeffs) - 1
        start_idx = deg - d
        for i, c in enumerate(coeffs):
            g_coeffs[start_idx + i] += scale * c
            
    add_poly(_H0, c0)
    add_poly(_H4, c1)
    add_poly(_H8, c2)
    add_poly(_H12, c3)
    
    # q(x) = g(x) / x^2
    # g_coeffs[12] is coeff of x^0. g_coeffs[11] is coeff of x^1.
    # Since g is even, g_coeffs[11] should be 0.
    # q(x) = sum_{k=2}^{12} g_k x^{k-2}
    # Coeff of x^{10} in q is coeff of x^{12} in g -> g_coeffs[0]
    # Coeff of x^0 in q is coeff of x^2 in g -> g_coeffs[10]
    # So q_coeffs = g_coeffs[0:11]
    
    q_coeffs = g_coeffs[0:11]
    
    # Find real roots of q
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return np.inf
    
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # Find sign-changing roots: q changes sign at the root
    # For a polynomial, a root is sign-changing if it has odd multiplicity
    # Numerically: check if q changes sign around the root
    sign_change_roots = []
    for r in real_roots:
        if abs(r) < 1e-15:
            continue
        # Check sign change: evaluate q at r-eps and r+eps
        eps = 1e-8 * max(1.0, abs(r))
        q_minus = np.polyval(q_coeffs, r - eps)
        q_plus = np.polyval(q_coeffs, r + eps)
        if q_minus * q_plus < 0:
            sign_change_roots.append(r)
    
    if not sign_change_roots:
        return np.inf
    
    r = max(sign_change_roots)
    
    # g tends to negative limit at infinity => replace g by -g
    # Leading coefficient of g is c3 * (leading of H12)
    # H12 leading coeff is 2^12 = 4096
    # If leading coeff of g < 0, g -> -inf, so we use -g
    # But the value formula uses r^2/(2*pi) regardless
    # Actually the task says "if g tends to a negative limit at infinity, g is replaced by -g"
    # This affects the sign of q and thus which roots are sign-changing
    # But r^2 is the same either way
    # So value = r^2 / (2*pi)
    
    return r * r / (2.0 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill
    
    # Seed
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_value(*seed)
    best_coeffs = seed.copy()
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    
    n_starts = 50
    # Generate diverse starting points
    starts = [seed.copy()]
    for i in range(n_starts - 1):
        # Random perturbations of varying scales
        scale = 0.1 * (10 ** (np.random.uniform(-2, 1)))
        starts.append(seed + scale * np.random.randn(3))
    
    # Also try some structured starts
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            start = seed.copy()
            start[0] += scale * np.cos(angle)
            start[1] += scale * np.sin(angle)
            starts.append(start)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.success or res.fun < best_val:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # High-precision polish on best found
    if best_val < np.inf and time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
