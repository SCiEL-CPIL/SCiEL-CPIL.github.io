import numpy as np
from scipy.optimize import differential_evolution
import sympy as sp

# Pre-compute Hermite polynomials coefficients
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 65536x^8 - 147456x^6 + 116128x^4 - 23040x^2 + 1344
# H12 = 16777216x^12 - 125829120x^10 + 331776000x^8 - 331776000x^6 + 119439360x^4 - 14745600x^2 + 479001

# Coefficients for x^0, x^2, ..., x^12
# H0: [1]
# H4: [12, -48, 16]
# H8: [1344, -23040, 116128, -147456, 65536]
# H12: [479001, -14745600, 119439360, -331776000, 331776000, -125829120, 16777216]

H_coeffs = {
    0: np.array([1.0]),
    4: np.array([12.0, -48.0, 16.0]),
    8: np.array([1344.0, -23040.0, 116128.0, -147456.0, 65536.0]),
    12: np.array([479001.0, -14745600.0, 119439360.0, -331776000.0, 331776000.0, -125829120.0, 16777216.0])
}

# Values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1344.0
H12_0 = 479001.0

def objective(params):
    c0, c1, c2 = params
    
    # Calculate c3
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Construct A_k coefficients for g(x) = sum A_k x^{2k}
    # A_k is the coefficient of x^{2k}
    # H0 contributes to A0
    # H4 contributes to A0, A1, A2
    # H8 contributes to A0..A4
    # H12 contributes to A0..A6
    
    A = np.zeros(7)
    
    # Add c0 * H0
    A[0] += c0 * H_coeffs[0][0]
    
    # Add c1 * H4
    A[0] += c1 * H_coeffs[4][0]
    A[1] += c1 * H_coeffs[4][1]
    A[2] += c1 * H_coeffs[4][2]
    
    # Add c2 * H8
    A[0] += c2 * H_coeffs[8][0]
    A[1] += c2 * H_coeffs[8][1]
    A[2] += c2 * H_coeffs[8][2]
    A[3] += c2 * H_coeffs[8][3]
    A[4] += c2 * H_coeffs[8][4]
    
    # Add c3 * H12
    A[0] += c3 * H_coeffs[12][0]
    A[1] += c3 * H_coeffs[12][1]
    A[2] += c3 * H_coeffs[12][2]
    A[3] += c3 * H_coeffs[12][3]
    A[4] += c3 * H_coeffs[12][4]
    A[5] += c3 * H_coeffs[12][5]
    A[6] += c3 * H_coeffs[12][6]
    
    # Check A[0] should be close to 0 due to c3 calculation
    # q(x) = g(x)/x^2 = A[1] + A[2]x^2 + ... + A[6]x^{10}
    # Let y = x^2. Q(y) = A[1] + A[2]y + A[3]y^2 + A[4]y^3 + A[5]y^4 + A[6]y^5
    
    Q_coeffs = A[1:7]
    
    # Find roots of Q(y)
    # Q_coeffs is [a5, a4, a3, a2, a1, a0] for np.roots
    # Wait, A[1:7] is [A1, A2, A3, A4, A5, A6]
    # Q(y) = A6*y^5 + A5*y^4 + A4*y^3 + A3*y^2 + A2*y + A1
    # np.roots expects highest degree first.
    # So we need [A6, A5, A4, A3, A2, A1]
    Q_poly_coeffs = A[6:0:-1]
    
    if np.allclose(Q_poly_coeffs, 0):
        return 1e10
        
    roots = np.roots(Q_poly_coeffs)
    
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-9:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10
        
    real_roots.sort()
    
    # Find largest root with sign change
    # We check sign of Q(y) at y_r - eps and y_r + eps
    max_y = 0
    for y_r in reversed(real_roots):
        delta = 1e-7 * max(1.0, y_r)
        y_minus = y_r - delta
        y_plus = y_r + delta
        
        if y_minus < 1e-9:
            continue
            
        # Evaluate Q(y)
        # Use np.polyval with Q_poly_coeffs
        q_minus = np.polyval(Q_poly_coeffs, y_minus)
        q_plus = np.polyval(Q_poly_coeffs, y_plus)
        
        if q_minus * q_plus < 0:
            max_y = y_r
            break
            
    if max_y == 0:
        return 1e10
        
    # Value is r^2 / (2 * pi)
    # We want to minimize this, so minimizing max_y is sufficient
    return max_y

def construct():
    # Seed coefficients
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    # Define bounds for differential evolution
    # Seed is approx [-1.13, 0.04, 0.0003]
    # We search in a wider range
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.1, 0.1)]
    
    # Run differential evolution
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        tol=1e-9, 
        popsize=15
    )
    
    if result.fun < 1e9: # Valid solution found
        return result.x
    else:
        return np.array([c0_seed, c1_seed, c2_seed])
