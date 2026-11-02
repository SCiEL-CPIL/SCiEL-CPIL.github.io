import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time
import scipy.fft as fft

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 175.0  # Keep margin for safety
    
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
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return g(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    def compute_A_f(c0, c4, c8):
        """Compute A(f) = integral |f(x)| dx from -inf to inf"""
        try:
            val, _ = integrate.quad(lambda x: abs(f_of_x(x, c0, c4, c8)), 0, np.inf, limit=200)
            return 2 * val
        except Exception:
            return np.inf
    
    def compute_A_F(c0, c4, c8):
        """Compute A(F(f)) = integral |F(f)(xi)| dxi from -inf to inf
        Using the property that Fourier transform of H_n(x)*exp(-x^2/2) is 
        (up to a constant factor and sign) H_n(x)*exp(-x^2/2)
        
        With the convention F[f](xi) = 1/sqrt(2*pi) * integral f(x)*exp(-i*x*xi) dx
        The Fourier transform of exp(-x^2/2) is exp(-xi^2/2)
        The Fourier transform of H_n(x)*exp(-x^2/2) is i^n * H_n(x)*exp(-x^2/2) (for physicist's Hermite polynomials)
        
        So F(f)(xi) = 1/sqrt(2*pi) * [c0*H0(xi) + i^4*c4*H4(xi) + i^8*c8*H8(xi) + i^12*c12*H12(xi)] * exp(-xi^2/2)
        Since i^4 = 1, i^8 = 1, i^12 = -1:
        F(f)(xi) = 1/sqrt(2*pi) * [c0*H0(xi) + c4*H4(xi) + c8*H8(xi) - c12*H12(xi)] * exp(-xi^2/2)
        """
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        # F(f)(xi) = 1/sqrt(2*pi) * [c0*H0(xi) + c4*H4(xi) + c8*H8(xi) - c12*H12(xi)] * exp(-xi^2/2)
        def F_f(xi):
            poly_val = c0*H0(xi) + c4*H4(xi) + c8*H8(xi) - c12*H12(xi)
            return poly_val * np.exp(-xi**2 / 2.0)
        
        try:
            val, _ = integrate.quad(lambda xi: abs(F_f(xi)), 0, np.inf, limit=200)
            return 2 * val / np.sqrt(2 * np.pi)
        except Exception:
            return np.inf
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        Af = compute_A_f(c0, c4, c8)
        if np.isinf(Af):
            return np.inf
        AF = compute_A_F(c0, c4, c8)
        if np.isinf(AF):
            return np.inf
        return Af * AF

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    current_time = time.time()
    
    # Primary optimization
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    current_time = time.time()
    
    # Random restarts if time permits
    np.random.seed(42)
    max_restarts = 20
    for i in range(max_restarts):
        if current_time > deadline - 15:
            break
        
        # Try different scales of perturbation
        scale = 0.1 * (0.5 ** (i // 5))
        noise = np.random.normal(0, scale, 3)
        x_try = best_coeffs + noise
        
        try:
            res = opt.minimize(objective, x_try, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        current_time = time.time()
    
    # Final polish with tighter tolerance if time permits
    if current_time < deadline - 30:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
