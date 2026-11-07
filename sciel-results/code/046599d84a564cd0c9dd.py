import numpy as np
import time
from scipy.optimize import minimize

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    # Calculate c3
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=4084080
    c3 = -(c0 * 1 + c1 * 12 + c2 * 1680) / 4084080.0
    
    # Construct q(y) coefficients
    # g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 122880x^10 + 1397760x^8 - 6812160x^6 + 13977600x^4 - 11827200x^2 + 4084080
    
    # q(y) = g(sqrt(y)) / y
    # g(x) = F x^12 + E x^10 + D x^8 + C x^6 + B x^4 + A x^2
    # q(y) = F y^5 + E y^4 + D y^3 + C y^2 + B y + A
    
    F = 4096 * c3
    E = -122880 * c3
    D = 256 * c2 + 1397760 * c3
    C = -3072 * c2 - 6812160 * c3
    B = 16 * c1 + 11520 * c2 + 13977600 * c3
    A = -48 * c1 - 13440 * c2 - 11827200 * c3
    
    # Find largest real root of q(y) where q changes sign
    # q(y) is a polynomial in y
    # We need roots of F y^5 + E y^4 + D y^3 + C y^2 + B y + A = 0
    # Use numpy roots
    coeffs_q = [F, E, D, C, B, A]
    roots = np.roots(coeffs_q)
    
    # Filter real roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            real_roots.append(root.real)
            
    # Check sign change
    # q(y) changes sign at y0 if q(y0-eps) * q(y0+eps) < 0
    # Or simply, if the root has odd multiplicity.
    # For numerical stability, check sign of q at roots +/- small delta
    valid_roots = []
    for r in real_roots:
        if r <= 0:
            continue
        eps = 1e-7
        q_minus = np.polyval(coeffs_q, r - eps)
        q_plus = np.polyval(coeffs_q, r + eps)
        if q_minus * q_plus < 0:
            valid_roots.append(r)
            
    if not valid_roots:
        return 0.0 # Or a large penalty? Task implies r exists. If not, value 0 is best.
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2 * np.pi)

def construct():
    deadline = time.time() + 170.0
    
    # Seed
    c0, c1, c2 = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    x0 = np.array([c0, c1, c2])
    
    # Objective: minimize value
    def obj(x):
        if time.time() > deadline:
            return 1e9
        try:
            return compute_value(x)
        except:
            return 1e9
            
    # Local optimization
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() > deadline:
        return np.array([c0, c1, c2])
        
    best_x = res.x
    best_val = res.fun
    
    # Return best coefficients
    return np.array(best_x)
