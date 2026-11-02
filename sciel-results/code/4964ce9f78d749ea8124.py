"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time


def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 73728x^10 + 580608x^8 - 2419200x^6 + 5443200x^4 - 5443200x^2 + 2032128
    
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return 4096*x**12 - 73728*x**10 + 580608*x**8 - 2419200*x**6 + 5443200*x**4 - 5443200*x**2 + 2032128
    
    def g(x, c0, c4, c8):
        # Constraint: g(0) = 0
        # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 2032128
        c12 = -(c0 + 12*c4 + 1680*c8) / 2032128.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f(x, c0, c4, c8):
        """Test function: g(x) * exp(-x^2/2)"""
        return g(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    def objective(coeffs):
        """Compute the uncertainty product Δx * Δp for the test function."""
        c0, c4, c8 = coeffs
        
        # Check deadline
        if time.time() - start_time > deadline:
            return np.inf
        
        # Compute L2 norm squared: ∫|f(x)|²dx
        norm_sq, _ = quad(lambda x: f(x, c0, c4, c8)**2, -np.inf, np.inf, limit=100)
        if norm_sq <= 0 or np.isnan(norm_sq):
            return np.inf
        
        # Compute second moment: ∫x²|f(x)|²dx
        moment_x, _ = quad(lambda x: x**2 * f(x, c0, c4, c8)**2, -np.inf, np.inf, limit=100)
        if np.isnan(moment_x):
            return np.inf
        
        # Variance in x: Δx² = moment_x / norm_sq
        var_x = moment_x / norm_sq
        
        # For Fourier-invariant functions (even Hermite polynomials), 
        # the Fourier transform has the same form, so Δp² = Δx²
        # Thus the uncertainty product is Δx * Δp = Δx²
        return var_x
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    # Try Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        if res.fun < best_score and not np.isnan(res.fun):
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Try random restarts if time permits
    np.random.seed(42)
    for i in range(20):
        if time.time() - start_time > deadline - 5:
            break
        
        # Perturb around best found
        noise = np.random.normal(0, 0.05, 3)
        x_try = best_coeffs + noise
        
        try:
            res = minimize(objective, x_try, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < best_score and not np.isnan(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 2:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000})
            if res.fun < best_score and not np.isnan(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
