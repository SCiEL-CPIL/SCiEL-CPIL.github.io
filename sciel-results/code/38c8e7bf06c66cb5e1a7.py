import numpy as np
from scipy.optimize import minimize
import time

def compute_P_coeffs(c0, c1, c2):
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 73728x^10 + 491520x^8 - 1428480x^6 + 1881600x^4 - 904320x^2 + 103950
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(0) = c0 + 12c1 + 1680c2 + 103950c3 = 0
    c3 = -(c0 + 12*c1 + 1680*c2) / 103950.0
    
    # g(x) is sum of even powers.
    # g(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2
    # q(x) = g(x)/x^2 = a12 x^10 + a10 x^8 + a8 x^6 + a6 x^4 + a4 x^2 + a2
    # Let y = x^2. P(y) = a12 y^5 + a10 y^4 + a8 y^3 + a6 y^2 + a4 y + a2
    
    # Coefficients of x^k in H_{2n}:
    # H0: x^0: 1
    # H4: x^4: 16, x^2: -48, x^0: 12
    # H8: x^8: 256, x^6: -3072, x^4: 11520, x^2: -13440, x^0: 1680
    # H12: x^12: 4096, x^10: -73728, x^8: 491520, x^6: -1428480, x^4: 1881600, x^2: -904320, x^0: 103950
    
    # a12 = c3 * 4096
    # a10 = c3 * (-73728)
    # a8 = c2 * 256 + c3 * 491520
    # a6 = c1 * 16 + c2 * (-3072) + c3 * (-1428480)  -- Wait, H4 has x^4, H8 has x^6?
    # H4: 16x^4. So x^4 coeff is 16.
    # H8: 256x^8 - 3072x^6... So x^6 coeff is -3072.
    # H12: ... -1428480x^6... So x^6 coeff is -1428480.
    # Correct.
    
    a12 = c3 * 4096.0
    a10 = c3 * (-73728.0)
    a8 = c2 * 256.0 + c3 * 491520.0
    a6 = c1 * 16.0 + c2 * (-3072.0) + c3 * (-1428480.0)
    a4 = c1 * (-48.0) + c2 * 11520.0 + c3 * 1881600.0
    a2 = c0 + c1 * 12.0 + c2 * (-13440.0) + c3 * (-904320.0)
    
    # P(y) = a12 y^5 + a10 y^4 + a8 y^3 + a6 y^2 + a4 y + a2
    return np.array([a12, a10, a8, a6, a4, a2])

def get_value(c0, c1, c2):
    coeffs = compute_P_coeffs(c0, c1, c2)
    # Find roots of P(y)
    roots = np.roots(coeffs)
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-12:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10 # Large penalty if no positive real roots (should not happen for valid g)
        
    # Check for sign change (simple root)
    # P(y) = sum coeffs[i] y^(5-i)
    # We can evaluate P at y +/- eps
    valid_roots = []
    for y in real_roots:
        # Check if it's a simple root
        # P(y) should be close to 0
        # P'(y) should be non-zero
        # Derivative coeffs: 5*a12 y^4 + 4*a10 y^3 + 3*a8 y^2 + 2*a6 y + a4
        d_coeffs = np.array([5*coeffs[0], 4*coeffs[1], 3*coeffs[2], 2*coeffs[3], coeffs[4]])
        deriv_val = np.polyval(d_coeffs, y)
        if abs(deriv_val) > 1e-6: # Simple root
            valid_roots.append(y)
            
    if not valid_roots:
        return 1e10
        
    max_y = max(valid_roots)
    return max_y / (2 * np.pi)

def construct():
    # Seed
    c_seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Objective function for optimizer
    def obj(c):
        return get_value(c[0], c[1], c[2])
        
    start_time = time.time()
    deadline = 180.0 # 180s limit, keep margin
    
    best_c = c_seed
    best_val = obj(best_c)
    
    # Multi-start Nelder-Mead
    # Try a few random starts and the seed
    starts = [c_seed]
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.uniform(-2, 2, size=3))
        
    for s in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(obj, s, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x
        except Exception:
            pass
            
    # If time remains, try Powell for higher precision
    if time.time() < deadline - 30:
        try:
            res = minimize(obj, best_c, method='Powell', options={'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
            
    return np.array(best_c)
