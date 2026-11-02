import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 180.0  # 180 seconds as per task description, keep margin for safety
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def g(x, c0, c4, c8):
        # Constraint g(0) = 0
        # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # H0(0) = 1
        # H4(0) = 3
        # H8(0) = 105
        # H12(0) = 10395
        # c12 = -(c0 + 3*c4 + 105*c8) / 10395
        
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        # We want to minimize A(g)^2, which is (integral |g(x) exp(-x^2/2) dx)^2
        # Actually, the task says "Hermite-polynomial times Gaussian test function".
        # Usually, the test function is f(x) = g(x) exp(-x^2/2).
        # Let's assume the function is f(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) * exp(-x^2/2).
        
        def f(x):
            return g(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
        
        # Integrate |f(x)| from -inf to inf
        # Since f(x) is even, integrate from 0 to inf and multiply by 2
        # Use quad for accuracy
        try:
            val, err = integrate.quad(lambda x: abs(f(x)), 0, np.inf, limit=100)
            return (2 * val) ** 2
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0
    
    # Try a few different optimization methods and initial points
    # Nelder-Mead is robust for non-smooth functions (absolute value)
    # BFGS/L-BFGS-B require gradients, which are tricky with abs()
    
    methods = ['Nelder-Mead']
    # We can also try perturbing the initial guess
    
    current_time = time.time()
    for method in methods:
        if current_time > deadline - 10:
            break
            
        try:
            res = opt.minimize(objective, x0, method=method, options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
        
        current_time = time.time()
        
        # Try a few random restarts if time permits
        np.random.seed(42)
        for i in range(10):
            if current_time > deadline - 10:
                break
            # Perturb around the best found so far
            noise = np.random.normal(0, 0.1, 3)
            x_try = best_coeffs + noise
            try:
                res = opt.minimize(objective, x_try, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
                if res.success and res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
            except Exception:
                pass
            current_time = time.time()

    # Return the coefficients c0, c4, c8
    return np.array(best_coeffs)
