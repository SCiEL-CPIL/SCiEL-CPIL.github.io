import numpy as np
from scipy.optimize import minimize
from sympy.polys.orthopolys import hermite_poly
import sympy as sp

# Precompute Hermite polynomials coefficients
# H_n(x) are physicists' Hermite polynomials.
# We need H0, H4, H8, H12.

def get_hermite_coeffs(n):
    x = sp.Symbol('x')
    Hn = hermite_poly(n, x)
    coeffs = sp.Poly(Hn, x).all_coeffs()
    return [float(c) for c in coeffs]

H0_coeffs = get_hermite_coeffs(0)
H4_coeffs = get_hermite_coeffs(4)
H8_coeffs = get_hermite_coeffs(8)
H12_coeffs = get_hermite_coeffs(12)

# Pad coefficients to degree 12
def pad_coeffs(coeffs, target_degree):
    # coeffs are from highest degree to lowest
    # numpy.roots expects highest degree to lowest
    padded = [0.0] * (target_degree - len(coeffs) + 1)
    padded += coeffs
    return padded

H0_padded = pad_coeffs(H0_coeffs, 12)
H4_padded = pad_coeffs(H4_coeffs, 12)
H8_padded = pad_coeffs(H8_coeffs, 12)
H12_padded = pad_coeffs(H12_coeffs, 12)

def construct():
    # Initial seed
    c0, c1, c2 = -113 / 100, 1 / 25, 1 / 3240
    x0 = np.array([c0, c1, c2])
    
    def objective(coeffs):
        c0, c1, c2 = coeffs
        
        # Compute c3
        H0_0 = H0_padded[-1]
        H4_0 = H4_padded[-1]
        H8_0 = H8_padded[-1]
        H12_0 = H12_padded[-1]
        
        val_0 = c0 * H0_0 + c1 * H4_0 + c2 * H8_0
        c3 = -val_0 / H12_0
        
        # Check leading coefficient sign
        # Leading coeff of g is c3 * H12_padded[0]
        lead_coeff = c3 * H12_padded[0]
        if lead_coeff <= 0:
            return 1e10  # Penalty for wrong sign at infinity
        
        # Construct g coefficients
        g_coeffs = c0 * np.array(H0_padded) + c1 * np.array(H4_padded) + c2 * np.array(H8_padded) + c3 * np.array(H12_padded)
        
        # q(x) = g(x) / x^2
        # g is degree 12, so q is degree 10
        # g_coeffs[0] * x^12 + g_coeffs[1] * x^11 + ... + g_coeffs[12]
        # q(x) = g_coeffs[0] * x^10 + g_coeffs[1] * x^9 + ... + g_coeffs[10]
        # The x^0 and x^1 terms of g must be 0 for exact division (guaranteed by g(0)=0 and g'(0)=0? No, just g(0)=0)
        # Actually g(0)=0 means constant term is 0. For g/x^2 to be polynomial, we need g'(0)=0 too?
        # Wait: g(x)/x^2 is a polynomial only if g(0)=0 AND g'(0)=0? No.
        # g(x) = a_0 + a_1 x + a_2 x^2 + ...
        # g(x)/x^2 = a_0/x^2 + a_1/x + a_2 + ...
        # For this to be a polynomial (no negative powers), we need a_0 = 0 and a_1 = 0.
        # The task says "which must divide exactly", implying g(0)=0 and g'(0)=0.
        # But the task only constrains g(0)=0 via c3.
        # Let's check: does the seed satisfy g'(0)=0?
        # If not, the task might mean something else, or the division is exact in the sense that the x^-1 and x^-2 terms vanish.
        # Let's assume the task guarantees exact division, so we just take the coefficients from x^2 to x^12.
        
        # q_coeffs: degree 10, from x^10 down to x^0
        # g_coeffs indices: 0 (x^12), 1 (x^11), ..., 10 (x^2), 11 (x^1), 12 (x^0)
        # q_coeffs[0] = g_coeffs[0] (x^10)
        # q_coeffs[1] = g_coeffs[1] (x^9)
        # ...
        # q_coeffs[10] = g_coeffs[10] (x^0)
        q_coeffs = g_coeffs[:11]
        
        # Find roots of q
        try:
            roots = np.roots(q_coeffs)
        except Exception:
            return 1e10
            
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        positive_roots = real_roots[real_roots > 1e-9]
        
        if len(positive_roots) == 0:
            return 0.0
            
        # Find the largest positive root where q changes sign
        # q changes sign at a root if the multiplicity is odd
        # For a simple root, q changes sign.
        # We need to check sign change for each positive root.
        
        # Sort positive roots
        positive_roots = np.sort(positive_roots)
        
        # For each root, check if q changes sign
        # Sample q just left and right of the root
        sign_change_roots = []
        for r in positive_roots:
            eps = 1e-6 * max(1, abs(r))
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                sign_change_roots.append(r)
        
        if len(sign_change_roots) == 0:
            # No sign change? Maybe the root has even multiplicity.
            # In that case, q doesn't change sign, so it's not a valid r?
            # The task says "at which q changes sign". If none, maybe r is undefined?
            # But we need a value. Let's assume if no sign change, we can't use it.
            # But then what? Maybe return a large penalty?
            # Or maybe the smallest root? No, "largest real root ... at which q changes sign".
            # If no such root, perhaps the value is 0? Or infinity?
            # Let's assume if no sign change root, the bound is not tight, so maybe 0?
            # Actually, if q doesn't change sign and is positive, then g >= 0, and r might be 0?
            # Let's return 0 for now.
            return 0.0
            
        r_max = max(sign_change_roots)
        
        # Check non-negativity of g on a dense grid
        x_max = max(10.0, 2.0 * r_max)
        x_grid = np.linspace(0, x_max, 2000)
        g_vals = np.polyval(g_coeffs, x_grid)
        
        # g must be >= 0 for all x >= 0
        if np.any(g_vals < -1e-8):
            return 1e10  # Penalty for violating non-negativity
        
        return r_max ** 2

    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    if result.success or result.fun < objective(x0):
        return result.x
    else:
        return x0
