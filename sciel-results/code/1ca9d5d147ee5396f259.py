"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Values at x=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

# We define q(x) = A(x)c0 + B(x)c1 + C(x)c2
# A(x) = (1 - H12(x)/18480) / x^2
# B(x) = (H4(x) - H12(x)/1540) / x^2
# C(x) = (H8(x) - H12(x)/110) / x^2

# Let's compute coefficients for A, B, C as arrays [x^10, x^9, ..., x^0]
# H12(x)/18480 = (4096/18480)x^12 - (46080/18480)x^10 + ...
# 1 - H12/18480 = 1 - [ ... + 18480/18480 ] = - [ ... ] (constant term cancels)
# The term x^12 in H12/18480 is 4096/18480.
# So A(x) has a term -4096/18480 x^10.

# Let's build the polynomials explicitly.
# H12_coeffs = [4096, 0, -46080, 0, 332160, 0, -1003520, 0, 1424640, 0, -921600, 0, 18480]
# Indices: 0:x^12, 1:x^11, ..., 12:x^0

def get_poly_coeffs(poly_type):
    """
    Returns coefficients of A, B, C from x^10 down to x^0.
    """
    # H12 / 18480
    h12_div_18480 = np.array([4096.0, 0, -46080.0, 0, 332160.0, 0, -1003520.0, 0, 1424640.0, 0, -921600.0, 0, 18480.0]) / 18480.0
    # H12 / 1540
    h12_div_1540 = np.array([4096.0, 0, -46080.0, 0, 332160.0, 0, -1003520.0, 0, 1424640.0, 0, -921600.0, 0, 18480.0]) / 1540.0
    # H12 / 110
    h12_div_110 = np.array([4096.0, 0, -46080.0, 0, 332160.0, 0, -1003520.0, 0, 1424640.0, 0, -921600.0, 0, 18480.0]) / 110.0
    
    # A(x) numerator: 1 - H12/18480
    # 1 is x^0. H12/18480 has x^0 term 1. So x^0 term is 0.
    # We need to divide by x^2.
    # Coeffs of 1 - H12/18480 (x^12 to x^0):
    num_A = np.zeros(13)
    num_A[12] = 1.0 # Constant 1
    num_A -= h12_div_18480
    
    # Divide by x^2: shift coeffs right by 2 (or take [2:] if we align to x^10)
    # num_A is [x^12, x^11, ..., x^0].
    # A(x) is degree 10.
    # Coeff of x^10 in A is num_A[2] (which was x^12 in num).
    # Wait, num_A[0] is x^12. num_A[2] is x^10.
    # A(x) = sum_{k=0}^{10} a_k x^k.
    # a_k = num_A[12-k] / x^2? No.
    # num_A(x) = sum_{j=0}^{12} c_j x^{12-j}.
    # A(x) = num_A(x) / x^2 = sum_{j=2}^{12} c_j x^{10-j}.
    # So A_coeffs[i] = num_A[i+2] for i=0..10.
    
    A_coeffs = num_A[2:13] # Length 11
    
    # B(x) numerator: H4 - H12/1540
    # H4 = 16x^4 - 48x^2 + 12
    num_B = np.zeros(13)
    # H4 coeffs: x^4: 16, x^2: -48, x^0: 12
    num_B[8] += 16.0 # x^4 is index 12-4=8
    num_B[10] += -48.0 # x^2 is index 12-2=10
    num_B[12] += 12.0 # x^0 is index 12
    num_B -= h12_div_1540
    
    B_coeffs = num_B[2:13]
    
    # C(x) numerator: H8 - H12/110
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    num_C = np.zeros(13)
    num_C[4] += 256.0 # x^8
    num_C[6] += -1344.0 # x^6
    num_C[8] += 2688.0 # x^4
    num_C[10] += -1344.0 # x^2
    num_C[12] += 168.0 # x^0
    num_C -= h12_div_110
    
    C_coeffs = num_C[2:13]
    
    if poly_type == 'A': return A_coeffs
    if poly_type == 'B': return B_coeffs
    if poly_type == 'C': return C_coeffs

A_coeffs = get_poly_coeffs('A')
B_coeffs = get_poly_coeffs('B')
C_coeffs = get_poly_coeffs('C')

def evaluate_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) where r is the largest real root of q where q changes sign."""
    # q(x) = A(x)c0 + B(x)c1 + C(x)c2
    q_coeffs = A_coeffs * c0 + B_coeffs * c1 + C_coeffs * c2
    
    # Check if leading coeff is 0, trim
    while len(q_coeffs) > 1 and abs(q_coeffs[0]) < 1e-12:
        q_coeffs = q_coeffs[1:]
        
    if len(q_coeffs) == 0:
        return float('inf')
        
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return float('inf')
    
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            real_roots.append(root.real)
    
    if not real_roots:
        return float('inf')
    
    sign_change_roots = []
    for r in real_roots:
        if r <= 0:
            continue
        
        # Evaluate q at r - eps and r + eps
        # Use small eps relative to r
        eps = 1e-7 * max(1.0, abs(r))
        q_before = np.polyval(q_coeffs, r - eps)
        q_after = np.polyval(q_coeffs, r + eps)
        
        if q_before * q_after < 0:
            sign_change_roots.append(r)
    
    if not sign_change_roots:
        return float('inf')
    
    r = max(sign_change_roots)
    value = r**2 / (2 * np.pi)
    
    return value

def construct():
    """Search for optimal coefficients c0, c1, c2."""
    start_time = time.time()
    deadline = start_time + 180 # Safety margin
    
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_value = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    # Strategy 1: Tangency Sweep
    # We want q(r) = 0 and q'(r) = 0.
    # q(x) = A(x)c0 + B(x)c1 + C(x)c2.
    # Set c2 = 1. Solve for c0, c1.
    # A(r)c0 + B(r)c1 + C(r) = 0
    # A'(r)c0 + B'(r)c1 + C'(r) = 0
    
    # Precompute derivatives? Or just use np.polyder.
    A_der = np.polyder(A_coeffs)
    B_der = np.polyder(B_coeffs)
    C_der = np.polyder(C_coeffs)
    
    # Sweep r
    r_values = np.linspace(0.1, 5.0, 200)
    
    for r in r_values:
        if time.time() > deadline - 30:
            break
            
        Ar = np.polyval(A_coeffs, r)
        Br = np.polyval(B_coeffs, r)
        Cr = np.polyval(C_coeffs, r)
        
        Apr = np.polyval(A_der, r)
        Bpr = np.polyval(B_der, r)
        Cpr = np.polyval(C_der, r)
        
        # Matrix:
        # [ Ar, Br ] [c0]   [ -Cr ]
        # [ Apr, Bpr] [c1] = [ -Cpr]
        
        M = np.array([[Ar, Br], [Apr, Bpr]])
        b = np.array([-Cr, -Cpr])
        
        det = np.linalg.det(M)
        if abs(det) < 1e-10:
            continue
            
        try:
            sol = np.linalg.solve(M, b)
            c0, c1 = sol
            c2 = 1.0
            
            # Check validity
            val = evaluate_value(c0, c1, c2)
            if val < best_value:
                best_value = val
                best_c = np.array([c0, c1, c2])
        except Exception:
            pass
    
    # Strategy 2: Local Refinement with Nelder-Mead
    def objective(coeffs):
        return evaluate_value(coeffs[0], coeffs[1], coeffs[2])
    
    try:
        result = minimize(
            objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result.fun < best_value and time.time() < deadline:
            best_c = result.x
            best_value = result.fun
    except Exception:
        pass
    
    # Strategy 3: Multi-start Nelder-Mead from diverse points
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 10:
            break
            
        # Random perturbation around best
        scale = 0.5 * (1 - i/10.0) + 0.1
        perturbation = np.random.randn(3) * scale
        candidate = best_c + perturbation
        
        val = objective(candidate)
        if val < best_value:
            best_c = candidate
            best_value = val
            
        # Try local optimization from this candidate
        try:
            res = minimize(
                objective,
                candidate,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_value:
                best_c = res.x
                best_value = res.fun
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
