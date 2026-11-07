"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_q_coeffs(c0, c1, c2):
    """
    Compute coefficients of Q(y) = g(sqrt(y))/y for y=x^2.
    Q(y) is a polynomial in y of degree 5.
    Returns np.array [q0, q1, q2, q3, q4, q5] where Q(y) = q0 + q1*y + ... + q5*y^5
    """
    # H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=9240
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 168.0) / 9240.0
    
    # Coefficients of g(x) for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # H0: 1
    # H4: 12 - 48x^2 + 16x^4
    # H8: 168 - 3360x^2 + 4320x^4 - 1920x^6 + 256x^8
    # H12: 9240 - 554400x^2 + 1209600x^4 - 1120000x^6 + 403200x^8 - 57600x^10 + 4096x^12
    
    g2 = c1 * (-48.0) + c2 * (-3360.0) + c3 * (-554400.0)
    g4 = c1 * 16.0 + c2 * 4320.0 + c3 * 1209600.0
    g6 = c2 * (-1920.0) + c3 * (-1120000.0)
    g8 = c2 * 256.0 + c3 * 403200.0
    g10 = c3 * (-57600.0)
    g12 = c3 * 4096.0
    
    # q(x) = g2 + g4*x^2 + g6*x^4 + g8*x^6 + g10*x^8 + g12*x^10
    # Q(y) = g2 + g4*y + g6*y^2 + g8*y^3 + g10*y^4 + g12*y^5
    Q_coeffs = np.array([g2, g4, g6, g8, g10, g12])
    
    return Q_coeffs

def _max_positive_sign_changing_root(Q_coeffs):
    """
    Find the largest positive real root of Q(y) where the sign changes.
    Returns -1 if no such root exists.
    """
    # Handle case where leading coeff is zero (degree reduction)
    # np.roots handles this if we trim, but it's safer to just use np.roots on the array
    # np.roots expects highest degree first.
    # If Q_coeffs[5] is 0, it's degree < 5.
    
    # Trim trailing zeros (highest degree)
    coeffs = Q_coeffs[::-1]
    # Find first non-zero
    first_non_zero = 0
    for i in range(len(coeffs)):
        if abs(coeffs[i]) > 1e-12:
            first_non_zero = i
            break
    if first_non_zero == len(coeffs):
        # Zero polynomial
        return -1.0
    coeffs = coeffs[first_non_zero:]
    
    roots = np.roots(coeffs)
    
    max_root = -1.0
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-9:
            # Check sign change
            # Use a small epsilon
            eps = 1e-8 * max(1.0, r.real)
            # Evaluate Q at r-eps and r+eps
            # Use Horner's method or np.polyval
            q_neg = np.polyval(coeffs, r.real - eps)
            q_pos = np.polyval(coeffs, r.real + eps)
            
            # Check if signs are different
            if q_neg * q_pos < 0:
                if r.real > max_root:
                    max_root = r.real
                    
    return max_root

def _compute_value(c0, c1, c2):
    """
    Compute the value r^2 / (2*pi).
    """
    try:
        Q_coeffs = _get_q_coeffs(c0, c1, c2)
        max_root = _max_positive_sign_changing_root(Q_coeffs)
        
        if max_root < 0:
            # No positive sign-changing roots.
            # The problem implies r is the largest such root.
            # If none exist, r is effectively 0 (or the bound is trivial).
            # Assuming value 0 is the goal.
            return 0.0
            
        r = np.sqrt(max_root)
        return (r * r) / (2.0 * np.pi)
    except Exception:
        return float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(params):
        if time.time() > deadline:
            return float('inf')
        val = _compute_value(params[0], params[1], params[2])
        return val
        
    # 1. Local optimization from seed
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if time.time() < deadline:
            val = _compute_value(res.x[0], res.x[1], res.x[2])
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
    except Exception:
        pass
        
    # 2. Random restarts to escape local minima
    # We want to find a region where the max root is small (or 0).
    np.random.seed(42)
    for i in range(100):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random initial point
        # Scale: coefficients in seed are small.
        # Let's try a wider range.
        init = seed + np.random.randn(3) * 2.0
        
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if time.time() > deadline - 5.0:
                break
            val = _compute_value(res.x[0], res.x[1], res.x[2])
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
                
                # If we hit 0, we can stop early
                if best_val < 1e-10:
                    break
        except Exception:
            pass
            
    return best_c

# EVOLVE-BLOCK-END
