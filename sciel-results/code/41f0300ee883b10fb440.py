import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin
    
    # Hermite polynomials
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def get_c12(c0, c4, c8):
        # P(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
        return -(c0 + 3*c4 + 105*c8) / 10395.0

    def P(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)

    def objective(coeffs):
        c0, c4, c8 = coeffs
        if time.time() - start_time > deadline:
            return np.inf
            
        # We want to minimize Variance = D_x / N
        # N = integral P(x)^2 exp(-x^2) dx
        # D_x = integral x^2 P(x)^2 exp(-x^2) dx
        
        # Define integrands
        # Since P is even, P(x)^2 is even. exp(-x^2) is even.
        # Integrate from 0 to inf and multiply by 2.
        
        def integrand_N(x):
            px = P(x, c0, c4, c8)
            return px**2 * np.exp(-x**2)
        
        def integrand_Dx(x):
            px = P(x, c0, c4, c8)
            return x**2 * px**2 * np.exp(-x**2)
        
        try:
            # Use quad with high precision
            N_val, _ = integrate.quad(integrand_N, 0, np.inf, limit=100)
            Dx_val, _ = integrate.quad(integrand_Dx, 0, np.inf, limit=100)
            
            # Multiply by 2 for full range
            N_val *= 2
            Dx_val *= 2
            
            if N_val == 0:
                return np.inf
                
            return Dx_val / N_val
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0
    
    # Optimize
    # Powell is robust for this kind of smooth objective
    try:
        res = opt.minimize(objective, x0, method='Powell', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    # Try Nelder-Mead as well
    if time.time() - start_time < deadline:
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Try random restarts if time permits
    np.random.seed(42)
    for i in range(20):
        if time.time() - start_time > deadline - 5:
            break
        # Perturb around best
        noise = np.random.normal(0, 0.5, 3)
        x_try = best_coeffs + noise
        try:
            res = opt.minimize(objective, x_try, method='Powell', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return np.array(best_coeffs)
