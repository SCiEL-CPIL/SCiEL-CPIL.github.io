"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_hermite

def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    try:
        # Hermite polynomial values at 0
        # H0(0) = 1, H4(0) = 12, H8(0) = 1344, H12(0) = 209088
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1344.0
        H12_0 = 209088.0
        
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Build q(x) = g(x) / x^2 where g = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # We need to compute the coefficients of q(x)
        # H0 = 1
        # H4 = x^4 - 6x^2 + 3
        # H8 = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
        # H12 = x^12 - 84x^10 + 1680x^8 - 16800x^6 + 60480x^4 - 30240x^2 + 10395
        
        # g(x) = c0*(1) + c1*(x^4 - 6x^2 + 3) + c2*(x^8 - 28x^6 + 280x^4 - 840x^2 + 105) + c3*(x^12 - 84x^10 + 1680x^8 - 16800x^6 + 60480x^4 - 30240x^2 + 10395)
        
        # Coefficients of g(x) from degree 12 down to 0:
        # x^12: c3
        # x^11: 0
        # x^10: -84*c3
        # x^9: 0
        # x^8: 1680*c3 + c2
        # x^7: 0
        # x^6: -16800*c3 - 28*c2
        # x^5: 0
        # x^4: 60480*c3 + 280*c2 + c1
        # x^3: 0
        # x^2: -30240*c3 - 840*c2 - 6*c1
        # x^1: 0
        # x^0: 10395*c3 + 105*c2 + 3*c1 + c0
        
        # Since g(0) = 0, the x^0 coefficient should be 0 (by construction of c3)
        # q(x) = g(x) / x^2, so we shift coefficients up by 2:
        # q(x) has degree 10
        # x^10: c3
        # x^9: 0
        # x^8: -84*c3
        # x^7: 0
        # x^6: 1680*c3 + c2
        # x^5: 0
        # x^4: -16800*c3 - 28*c2
        # x^3: 0
        # x^2: 60480*c3 + 280*c2 + c1
        # x^1: 0
        # x^0: -30240*c3 - 840*c2 - 6*c1
        
        # For q to divide exactly, we need the x^0 and x^1 terms to be 0
        # x^1 is already 0
        # x^0: -30240*c3 - 840*c2 - 6*c1 = 0
        
        # Check if q divides exactly
        x0_coeff = -30240*c3 - 840*c2 - 6*c1
        if abs(x0_coeff) > 1e-10:
            return float('inf')
        
        # Build polynomial coefficients for q(x) (degree 10 down to 0)
        q_coeffs = np.array([
            c3,          # x^10
            0,           # x^9
            -84*c3,      # x^8
            0,           # x^7
            1680*c3 + c2, # x^6
            0,           # x^5
            -16800*c3 - 28*c2, # x^4
            0,           # x^3
            60480*c3 + 280*c2 + c1, # x^2
            0,           # x^1
            -30240*c3 - 840*c2 - 6*c1 # x^0
        ])
        
        # Find roots of q(x)
        roots = np.roots(q_coeffs)
        
        # Find real roots
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        
        # For each real root, check if q changes sign (odd multiplicity)
        # We need to find the largest real root where q changes sign
        best_r = None
        for r in real_roots:
            # Check sign change by evaluating q at r-eps and r+eps
            eps = 1e-8
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:  # Sign change
                if best_r is None or r > best_r:
                    best_r = r
        
        if best_r is None:
            return float('inf')
        
        value = best_r**2 / (2 * np.pi)
        return value
        
    except Exception:
        return float('inf')

def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2)

def construct():
    """Construct three coefficients that minimize the objective value."""
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed.copy()
    best_value = _compute_value(*seed)
    
    # Try different optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}),
        ('BFGS', {'maxiter': 500, 'gtol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 500, 'gtol': 1e-10}),
    ]
    
    for method, options in strategies:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(_objective, best_params, method=method, options=options, timeout=60)
            if result.success and result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            continue
    
    # Try local perturbations around the best solution
    if time.time() < deadline:
        base_value = _compute_value(*best_params)
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Random perturbation
            scale = 1e-3
            perturbation = np.random.normal(0, scale, size=3)
            candidate = best_params + perturbation
            
            val = _compute_value(*candidate)
            if val < best_value:
                best_params = candidate
                best_value = val
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
