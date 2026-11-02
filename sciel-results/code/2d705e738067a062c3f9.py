"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time


def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s limit
    
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
    
    def poly_deriv(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        # Derivatives: H0'=0, H4'=4x^3-12x, H8'=8x^7-168x^5+840x^3-840x, H12'=12x^11-660x^9+10080x^7-55440x^5+92400x^3-55440x
        dH0 = np.zeros_like(x, dtype=float)
        dH4 = 4*x**3 - 12*x
        dH8 = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        dH12 = 12*x**11 - 660*x**9 + 10080*x**7 - 55440*x**5 + 92400*x**3 - 55440*x
        return c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        # Check time
        if time.time() > deadline - 5:
            return np.inf
        
        # Compute integrals using quad
        # I0 = ∫P²e^(-x²)dx
        # Ix2 = ∫x²P²e^(-x²)dx
        # Ideriv = ∫(P'-xP)²e^(-x²)dx
        
        try:
            I0, _ = integrate.quad(lambda x: poly(x, c0, c4, c8)**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if I0 <= 0 or not np.isfinite(I0):
                return np.inf
            
            Ix2, _ = integrate.quad(lambda x: x**2 * poly(x, c0, c4, c8)**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
            
            Ideriv, _ = integrate.quad(lambda x: (poly_deriv(x, c0, c4, c8) - x*poly(x, c0, c4, c8))**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
            
            # A(f)² = Ix2 / I0
            # A(ĝ)² = (1/(4π²)) * Ideriv / I0
            # Product = (Ix2/I0) * (1/(4π²)) * (Ideriv/I0)
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    # Nelder-Mead optimization
    current_time = time.time()
    try:
        res = opt.minimize(compute_uncertainty_product, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'adaptive': True})
        if res.fun < best_score and np.isfinite(res.fun):
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    current_time = time.time()
    
    # Random restarts if time permits
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb around the best found so far
        noise_scale = 0.1 / (1 + i * 0.1)  # Decreasing perturbation
        noise = np.random.normal(0, noise_scale, 3)
        x_try = best_coeffs + noise
        try:
            res = opt.minimize(compute_uncertainty_product, x_try, method='Nelder-Mead', 
                              options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Return the coefficients c0, c4, c8
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
