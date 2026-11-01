import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

START_TIME = time.time()
TIME_LIMIT = 200.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        def integrand(x):
            h0 = 1.0
            h4 = eval_hermite(4, x)
            h8 = eval_hermite(8, x)
            h12 = eval_hermite(12, x)
            
            p_val = c0*h0 + c4*h4 + c8*h8 + c12*h12
            g_val = p_val * np.exp(-0.5 * x**2)
            
            return (x**4) * np.abs(g_val)
        
        from scipy.integrate import quad
        val, err = quad(integrand, 0, np.inf, limit=200)
        return 2.0 * val

    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score

    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    num_starts = 50
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 20:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
        except Exception:
            pass

    if time.time() > TIME_LIMIT - 5:
        return best_coeffs

    return best_coeffs
