import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

START_TIME = time.time()
TIME_LIMIT = 195.0  # Leave 15s safety margin for 210s total

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute Hermite polynomial coefficients for vectorized evaluation
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 10395
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def compute_p(x, c0, c4, c8, c12):
        """Vectorized computation of P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12"""
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        
        h0 = np.ones_like(x)
        h4 = x4 - 6.0 * x2 + 3.0
        h8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
        h12 = x12 - 84.0 * x10 + 2520.0 * x8 - 30240.0 * x6 + 151200.0 * x4 - 221760.0 * x2 + 10395.0
        
        return c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        def integrand(x):
            p_val = compute_p(x, c0, c4, c8, c12)
            g_val = p_val * np.exp(-0.5 * x**2)
            return (x**4) * np.abs(g_val)
        
        val, err = quad(integrand, 0, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
        return 2.0 * val

    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Phase 1: Multi-start Nelder-Mead
    num_starts = 40
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish with tighter tolerance
    if time.time() < TIME_LIMIT - 15:
        try:
            res2 = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < TIME_LIMIT - 5:
        try:
            res3 = minimize(objective, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x.copy()
        except Exception:
            pass

    return best_coeffs
