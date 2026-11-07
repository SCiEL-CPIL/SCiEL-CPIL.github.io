"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Constants for Hermite polynomial evaluations at 0
H0_0_val = 1.0
H4_0_val = 3.0
H8_0_val = 105.0
H12_0_val = 10395.0

def compute_value(c0, c1, c2):
    """Compute the value for given coefficients c0, c1, c2."""
    c3 = -(c0 * H0_0_val + c1 * H4_0_val + c2 * H8_0_val) / H12_0_val
    
    # Coefficients of Q(y) where y = x^2, Q(y) = g(x)/x^2
    # g(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395
    
    # g(x) = g0 + g2*x^2 + g4*x^4 + g6*x^6 + g8*x^8 + g10*x^10 + g12*x^12
    # g0 = 0 (by construction)
    g12 = c3
    g10 = -84 * c3
    g8 = 2100 * c3
    g6 = -25200 * c3 - 28 * c2
    g4 = 138600 * c3 + 210 * c2 + c1
    g2 = -332640 * c3 - 420 * c2 - 6 * c1
    
    # Q(y) = g12*y^5 + g10*y^4 + g8*y^3 + g6*y^2 + g4*y + g2
    coeffs_Q = [g12, g10, g8, g6, g4, g2]
    
    # Check if leading coefficient is zero (degenerate case)
    if abs(coeffs_Q[0]) < 1e-15:
        # Try to reduce degree
        non_zero = [i for i, c in enumerate(coeffs_Q) if abs(c) > 1e-15]
        if not non_zero:
            return 1e10
        max_deg = max(non_zero)
        coeffs_Q = coeffs_Q[:max_deg+1]
        if len(coeffs_Q) < 2:
            return 1e10
    
    # Find roots of Q(y)
    try:
        roots_y = np.roots(coeffs_Q)
    except:
        return 1e10
    
    # Filter for real, positive roots with sign change
    real_roots = []
    for root in roots_y:
        if abs(np.imag(root)) < 1e-8:
            r_val = np.real(root)
            if r_val > 1e-9:
                # Check sign change using multiple epsilon values
                found_sign_change = False
                for eps_factor in [1e-6, 1e-5, 1e-4, 1e-3]:
                    eps = eps_factor * max(1.0, abs(r_val))
                    y1 = r_val - eps
                    y2 = r_val + eps
                    if y1 <= 0:
                        continue
                    q1 = np.polyval(coeffs_Q, y1)
                    q2 = np.polyval(coeffs_Q, y2)
                    if q1 * q2 < 0:
                        found_sign_change = True
                        break
                if found_sign_change:
                    real_roots.append(r_val)
    
    if not real_roots:
        return 1e10
        
    R_max = max(real_roots)
    r_x = np.sqrt(R_max)
    value = (r_x ** 2) / (2 * np.pi)
    return value

def construct():
    """Find optimal coefficients using multi-start local optimization."""
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_val = compute_value(seed[0], seed[1], seed[2])
    
    best_value = seed_val
    best_coeffs = seed.copy()
    
    # Multi-start with random perturbations
    np.random.seed(42)
    
    # Define objective function
    def objective(coeffs):
        return compute_value(coeffs[0], coeffs[1], coeffs[2])
    
    # Try multiple starting points
    start_points = [seed]
    
    # Add random perturbations around seed
    for i in range(20):
        scale = 0.1 * (0.5 ** (i // 5))
        perturbation = np.random.randn(3) * scale
        start_points.append(seed + perturbation)
    
    # Add some larger random starts
    for i in range(10):
        start_points.append(np.random.randn(3) * 0.5)
    
    start_points = np.array(start_points)
    
    start_time = time.time()
    
    for sp in start_points:
        if time.time() - start_time > 150:  # Leave 30s buffer
            break
            
        try:
            res = minimize(objective, sp, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 1000})
            
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Final refinement with tighter tolerance
    if best_value < seed_val:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 2000})
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
