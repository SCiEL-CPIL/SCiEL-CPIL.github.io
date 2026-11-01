import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 195.0  # Safety margin before 210s

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Pre-compute Hermite polynomials at 0 for constraint
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Define the polynomial part P(x)
        # We need to integrate x^4 * |P(x) * exp(-x^2/2)| from 0 to inf
        # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        
        def integrand(x):
            h0 = 1.0
            h4 = eval_hermite(4, x)
            h8 = eval_hermite(8, x)
            h12 = eval_hermite(12, x)
            
            p_val = c0*h0 + c4*h4 + c8*h8 + c12*h12
            g_val = p_val * np.exp(-0.5 * x**2)
            
            return (x**4) * np.abs(g_val)
        
        # Integrate from 0 to infinity
        val, err = quad(integrand, 0, np.inf, limit=200)
        # Multiply by 2 for symmetry over -inf to inf
        return 2.0 * val

    # Initial guesses
    seed_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try the seed first
    seed_val = objective(best_coeffs)
    if seed_val < seed_score:
        seed_score = seed_val

    # Multi-start optimization with adaptive bounds
    np.random.seed(42)
    
    # Define bounds based on typical magnitudes
    bounds = [(-10.0, 10.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    num_starts = 80
    best_score = seed_score
    best_coeffs = best_coeffs.copy()
    
    # Generate diverse starting points
    start_points = []
    
    # Include the seed
    start_points.append(best_coeffs.copy())
    
    # Random points in log-spaced ranges around expected scales
    for _ in range(num_starts - 1):
        if time.time() > TIME_LIMIT - 30:
            break
        c0 = np.random.uniform(bounds[0][0], bounds[0][1])
        c4 = np.random.uniform(bounds[1][0], bounds[1][1])
        c8 = np.random.uniform(bounds[2][0], bounds[2][1])
        start_points.append(np.array([c0, c4, c8]))
    
    # Sort by distance from seed to prioritize nearby points first
    seed_vec = best_coeffs
    start_points.sort(key=lambda x: np.linalg.norm(x - seed_vec))
    
    for i, x0 in enumerate(start_points):
        if time.time() > TIME_LIMIT - 10:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth objective
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < TIME_LIMIT - 15:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
