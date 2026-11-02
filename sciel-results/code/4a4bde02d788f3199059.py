import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    # H_n(x) defined by H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        # Constraint P(0) = 0 => c0 + 3c4 + 105c8 + 10395 c12 = 0
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def poly_deriv(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        # Derivatives:
        # H0' = 0
        # H4' = 4x^3 - 12x
        # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
        # H12' = 12x^11 - 660x^9 + 10080x^7 - 55440x^5 + 92400x^3 - 55440x
        dH0 = np.zeros_like(x, dtype=float)
        dH4 = 4*x**3 - 12*x
        dH8 = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        dH12 = 12*x**11 - 660*x**9 + 10080*x**7 - 55440*x**5 + 92400*x**3 - 55440*x
        return c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        # Check time
        if time.time() > deadline - 2:
            return np.inf
        
        # Compute integrals using quad
        # I0 = ∫ P(x)^2 e^(-x^2) dx
        # Ix2 = ∫ x^2 P(x)^2 e^(-x^2) dx
        # Ideriv = ∫ (P'(x) - x P(x))^2 e^(-x^2) dx
        
        try:
            # Define integrands
            def integrand_I0(x):
                p = poly(x, c0, c4, c8)
                return p**2 * np.exp(-x**2)
            
            def integrand_Ix2(x):
                p = poly(x, c0, c4, c8)
                return x**2 * p**2 * np.exp(-x**2)
            
            def integrand_Ideriv(x):
                p = poly(x, c0, c4, c8)
                dp = poly_deriv(x, c0, c4, c8)
                term = dp - x * p
                return term**2 * np.exp(-x**2)
            
            # Perform integration
            # Using limit=200 for safety with oscillatory or complex-looking integrands, 
            # though these are smooth.
            I0, _ = integrate.quad(integrand_I0, -np.inf, np.inf, limit=200)
            if I0 <= 0 or not np.isfinite(I0):
                return np.inf
            
            Ix2, _ = integrate.quad(integrand_Ix2, -np.inf, np.inf, limit=200)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
            
            Ideriv, _ = integrate.quad(integrand_Ideriv, -np.inf, np.inf, limit=200)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
            
            # A(f)^2 = Ix2 / I0
            # A(hat{f})^2 = (1/(2*pi)^2) * Ideriv / I0
            # Product = sqrt( (Ix2/I0) * (Ideriv/I0) / (4*pi^2) )
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf
    
    # Initial guess from seed (Goncalves et al. 2017)
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    # Optimization Strategy:
    # 1. Nelder-Mead from seed
    # 2. Random restarts around seed and best found
    # 3. Polish with tighter tolerances
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local optimization from seed
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(compute_uncertainty_product, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Random restarts
    # The landscape might have multiple local minima.
    # We restart around the current best and the seed.
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate a starting point
        if i % 2 == 0:
            # Perturb around best
            scale = 0.1 * (1 + i/10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        else:
            # Perturb around seed
            scale = 0.2
            x_start = x0 + rng.normal(0, scale, 3)
            
        try:
            res = opt.minimize(compute_uncertainty_product, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 300, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
