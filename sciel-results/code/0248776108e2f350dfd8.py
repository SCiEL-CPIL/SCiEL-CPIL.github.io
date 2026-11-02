import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return poly(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # We need ∫|P(x)|*exp(-x²/2)dx = ∫|P(x)|*exp(x²/2)*exp(-x²)dx
    # Using Gauss-Hermite with weight exp(-x²):
    # ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
    # So ∫|P(x)|*exp(-x²/2)dx ≈ Σ w_i |P(x_i)|*exp(x_i²/2)
    n_gh = 100
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)
    
    def objective_fast(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            p_vals = poly(x_gh, c0, c4, c8)
            abs_f = np.abs(p_vals) * np.exp(x_gh**2 / 2.0)
            integral = np.sum(w_gh * abs_f)
            return (2 * integral) ** 2
        except Exception:
            return np.inf
    
    # Verify with quad for the initial guess
    def objective_quad(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            def abs_f(x):
                return abs(f_of_x(x, c0, c4, c8))
            val, _ = integrate.quad(abs_f, 0, np.inf, limit=200)
            return (2 * val) ** 2
        except Exception:
            return np.inf
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective_fast(x0)
    best_coeffs = x0.copy()
    
    # Try Nelder-Mead with multiple restarts using fast objective
    rng = np.random.RandomState(42)
    
    for restart in range(200):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            x_start = x0
        else:
            # Random restart around best with decreasing scale
            scale = 0.3 * (1 + 5.0/(restart + 1.0))
            x_start = best_coeffs + rng.normal(0, scale, 3)
        
        try:
            res = opt.minimize(objective_fast, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with high-precision quad
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(objective_quad, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # One more polish pass with even tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective_quad, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
