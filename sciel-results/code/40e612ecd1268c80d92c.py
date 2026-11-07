import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Hermite polynomial coefficients
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 4915200x^6 + 8709120x^4 - 5529600x^2 + 10321920

# Coefficients in order x^0, x^2, x^4, x^6, x^8, x^10, x^12 (even powers only)
H0 = np.array([1.0])
H4 = np.array([12.0, -48.0, 16.0])
H8 = np.array([1680.0, -13440.0, 11520.0, -3072.0, 256.0])
H12 = np.array([10321920.0, -5529600.0, 8709120.0, -4915200.0, 1032192.0, -98304.0, 4096.0])

def _get_q_coeffs(c0, c1, c2):
    """Get coefficients of q(x) = g(x)/x^2. Returns poly in x for np.roots (descending order)."""
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 10321920.0
    
    # g(x) coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    g = np.zeros(7)
    for i in range(len(H0)):
        g[i] += c0 * H0[i]
    for i in range(len(H4)):
        g[i] += c1 * H4[i]
    for i in range(len(H8)):
        g[i] += c2 * H8[i]
    for i in range(len(H12)):
        g[i] += c3 * H12[i]
    
    # q(x) = g(x)/x^2: coefficients for x^0, x^2, x^4, x^6, x^8, x^10
    q_even = g[1:]  # length 6
    
    # Convert to full polynomial in x (odd powers are 0)
    # q(x) = q_even[0] + q_even[1]*x^2 + q_even[2]*x^4 + q_even[3]*x^6 + q_even[4]*x^8 + q_even[5]*x^10
    poly_x = np.zeros(11)  # x^0 to x^10
    for i, coeff in enumerate(q_even):
        poly_x[2*i] = coeff
    
    # np.roots needs descending order
    poly_desc = poly_x[::-1]
    
    # Trim leading zeros
    while len(poly_desc) > 1 and abs(poly_desc[0]) < 1e-14:
        poly_desc = poly_desc[1:]
    
    return poly_desc

def _compute_value(c):
    """Compute r^2/(2*pi) where r is largest sign-changing real root of q."""
    c0, c1, c2 = c
    try:
        poly_desc = _get_q_coeffs(c0, c1, c2)
        
        if len(poly_desc) <= 1:
            return 1e10
        
        roots = np.roots(poly_desc)
        
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-7 * max(1.0, abs(r.real)):
                real_roots.append(r.real)
        
        if not real_roots:
            return 1e10
        
        sign_changing = []
        eps = 1e-6
        for r in real_roots:
            if abs(r) < eps:
                continue
            val_minus = np.polyval(poly_desc, r - eps)
            val_plus = np.polyval(poly_desc, r + eps)
            if val_minus * val_plus < 0:
                sign_changing.append(r)
        
        if not sign_changing:
            return 1e10
        
        r_max = max(sign_changing)
        return (r_max ** 2) / (2.0 * np.pi)
        
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_val = _compute_value(best_c)
    
    # Local refinement from seed
    try:
        result = minimize(_compute_value, seed, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_val:
            best_c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_c.tolist()
    
    # Differential evolution
    try:
        bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
        de_result = differential_evolution(
            _compute_value, bounds, seed=42, maxiter=150,
            popsize=12, tol=1e-6, mutation=(0.3, 1.0),
            recombination=0.7, init='sobol', polish=True
        )
        if de_result.fun < best_val:
            best_c = de_result.x
            best_val = de_result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_c.tolist()
    
    # Additional local refinements
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(_compute_value, best_c, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_val:
                best_c = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return best_c.tolist()
