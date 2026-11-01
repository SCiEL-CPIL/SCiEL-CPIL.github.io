import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def _hermite_polys(x):
    """Return H0, H4, H8, H12 evaluated at scalar x."""
    x2 = x*x
    H0 = 1.0
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 3072*x2*x2*x2 + 11520*x2*x2 - 13440*x2 + 1680
    H12 = (4096*x2*x2*x2*x2*x2*x2 - 73728*x2*x2*x2*x2*x2 + 
           580608*x2*x2*x2*x2 - 2125824*x2*x2*x2 + 
           3635520*x2*x2 - 2903040*x2 + 103950)
    return H0, H4, H8, H12

def _compute_a12(a0, a4, a8):
    return -(a0 + 12*a4 + 1680*a8) / 103950.0

def _objective(coeffs):
    a0, a4, a8 = coeffs
    a12 = _compute_a12(a0, a4, a8)
    
    # Denominator
    denom = np.sqrt(np.pi) * (
        a0**2 +
        a4**2 * 16 * 24 +
        a8**2 * 256 * 40320 +
        a12**2 * 4096 * 479001600
    )
    
    if denom <= 0:
        return 1e10
    
    # Integrand for L1 norm
    def integrand(x):
        H0, H4, H8, H12 = _hermite_polys(x)
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        return abs(P) * np.exp(-x*x/2)
    
    integral, _ = quad(integrand, -15, 15, limit=100, epsabs=1e-10, epsrel=1e-10)
    numerator = integral * integral
    
    return numerator / denom

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Leave margin
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try multiple starting points
    np.random.seed(42)
    n_starts = 10
    
    for i in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        if i == 0:
            x_start = x0
        else:
            # Random perturbation
            x_start = x0 + np.random.randn(3) * 0.5
        
        try:
            res = minimize(_objective, x_start, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return np.array(best_x)
