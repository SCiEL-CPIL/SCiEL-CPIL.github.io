import numpy as np
import scipy.optimize as so
import time
from scipy import roots as polyroots

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    # Hermite polynomial coefficients (physicists' convention)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 168x^4 - 168x^2 + 105
    # H12 = x^12 - 60x^10 + 1260x^8 - 5040x^6 + 30240x^4 - 30240x^2 + 945
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    c3 = -(c0 * 1 + c1 * 3 + c2 * 105) / 945.0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Coefficients of g(x) as polynomial in x (even powers only)
    # x^0: c0*1 + c1*3 + c2*105 + c3*945 = 0 (by construction)
    # x^2: c1*(-6) + c2*(-168) + c3*(-30240)
    # x^4: c1*1 + c2*168 + c3*30240
    # x^6: c2*(-28) + c3*(-5040)
    # x^8: c2*1 + c3*1260
    # x^10: c3*(-60)
    # x^12: c3*1
    
    a2 = c1 * (-6) + c2 * (-168) + c3 * (-30240)
    a4 = c1 * 1 + c2 * 168 + c3 * 30240
    a6 = c2 * (-28) + c3 * (-5040)
    a8 = c2 * 1 + c3 * 1260
    a10 = c3 * (-60)
    a12 = c3 * 1
    
    # q(x) = g(x)/x^2 = a2 + a4*x^2 + a6*x^4 + a8*x^6 + a10*x^8 + a12*x^10
    # Let z = x^2, p(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    # p(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    
    # Check if g tends to negative limit at infinity: leading coeff is a12 = c3
    # If c3 < 0, g -> -inf, so we replace g by -g, which negates all coefficients
    if a12 < 0:
        a2, a4, a6, a8, a10, a12 = -a2, -a4, -a6, -a8, -a10, -a12
    
    # Find roots of p(z) = a12*z^5 + a10*z^4 + a8*z^3 + a6*z^2 + a4*z + a2
    coeffs = [a12, a10, a8, a6, a4, a2]
    
    # Use numpy's polynomial root finding
    try:
        z_roots = np.roots(coeffs)
    except:
        return np.inf
    
    # Filter for real, positive roots
    real_positive_roots = []
    for zr in z_roots:
        if abs(zr.imag) < 1e-10:
            zr_real = zr.real
            if zr_real > 1e-12:
                real_positive_roots.append(zr_real)
    
    if not real_positive_roots:
        return np.inf
    
    # For each positive real root z, check if p changes sign there
    # p changes sign at a root if it's a root of odd multiplicity
    # Numerically: check sign of p(z - eps) and p(z + eps)
    def p_func(z):
        return a12*z**5 + a10*z**4 + a8*z**3 + a6*z**2 + a4*z + a2
    
    sign_changing_roots = []
    for zr in real_positive_roots:
        eps = 1e-8 * max(1, abs(zr))
        val_before = p_func(zr - eps)
        val_after = p_func(zr + eps)
        if val_before * val_after < 0:
            sign_changing_roots.append(zr)
    
    if not sign_changing_roots:
        return np.inf
    
    # r is the largest real root of q at which q changes sign
    # z = r^2, so r = sqrt(z)
    # Largest z gives largest r
    max_z = max(sign_changing_roots)
    r = np.sqrt(max_z)
    value = r**2 / (2 * np.pi)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 195  # 15s safety margin before 210s kill
    
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_c = seed.copy()
    best_val = _compute_value(seed[0], seed[1], seed[2])
    
    # Multi-start Nelder-Mead search
    np.random.seed(42)
    
    # Generate multiple starting points around the seed
    starts = [seed.copy()]
    for i in range(20):
        if time.time() > deadline - 30:
            break
        scale = 0.5 * (0.5 ** (i // 5))
        offset = np.random.randn(3) * scale
        start = seed + offset
        starts.append(start)
    
    # Also try some structured starts
    for c0 in [-1.5, -1.0, -0.5]:
        for c1 in [-0.2, 0.0, 0.1, 0.3]:
            for c2 in [0.0, 0.001, 0.005]:
                if time.time() > deadline - 30:
                    break
                starts.append(np.array([c0, c1, c2]))
    
    for start in starts:
        if time.time() > deadline - 15:
            break
        
        def objective(c):
            try:
                return _compute_value(c[0], c[1], c[2])
            except:
                return 1e10
        
        try:
            result = so.minimize(objective, start, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if result.fun < best_val and np.isfinite(result.fun):
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # High-precision polish on the best point found
    if time.time() < deadline - 5:
        def objective(c):
            try:
                return _compute_value(c[0], c[1], c[2])
            except:
                return 1e10
        
        try:
            result = so.minimize(objective, best_c, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_val and np.isfinite(result.fun):
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if not np.isfinite(final_val) or final_val > best_val:
        # Fall back to seed
        return seed
    
    return np.array([best_c[0], best_c[1], best_c[2]])
