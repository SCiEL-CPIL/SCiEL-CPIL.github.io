"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import time

# Hermite Polynomials (Physicists')
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H_12(x) = 4096x^12 - 122880x^10 + 1290240x^8 - 5308416x^6 + 9031680x^4 - 5308416x^2 + 1039560
# Note: H_n(0) for even n:
# H_0(0) = 1
# H_4(0) = 12
# H_8(0) = 1680
# H_12(0) = 1039560

def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680

def H12(x):
    return 4096*x**12 - 122880*x**10 + 1290240*x**8 - 5308416*x**6 + 9031680*x**4 - 5308416*x**2 + 1039560

def get_a12(a0, a4, a8):
    # f(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=1039560
    return -(a0*1.0 + a4*12.0 + a8*1680.0) / 1039560.0

def objective(coeffs):
    a0, a4, a8 = coeffs
    a12 = get_a12(a0, a4, a8)
    
    # Define f(x) and hat_f(w)
    # f(x) = exp(-x^2/2) * (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x))
    # hat_f(w) = sqrt(2*pi) * exp(-w^2/2) * (a0*H0(w) + a4*(-i)^4*H4(w) + a8*(-i)^8*H8(w) + a12*(-i)^12*H12(w))
    # Note: (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    # So hat_f(w) has the same polynomial structure but with w instead of x, multiplied by sqrt(2*pi).
    
    def f_abs_integrand(x):
        val = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * np.exp(-0.5*x*x)
        return np.abs(val)
        
    def hat_f_abs_integrand(w):
        val = (a0*H0(w) + a4*H4(w) + a8*H8(w) + a12*H12(w)) * np.exp(-0.5*w*w)
        return np.sqrt(2*np.pi) * np.abs(val)
        
    try:
        A_f, _ = quad(f_abs_integrand, -np.inf, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
        A_hat_f, _ = quad(hat_f_abs_integrand, -np.inf, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
        return A_f * A_hat_f
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin
    
    # Initial guess from seed
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = objective(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Differential Evolution
    # Bounds: The coefficients can vary. 
    # Based on seed: a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003.
    # Let's try wider bounds to allow for global search.
    bounds = [(-5, 5), (-5, 5), (-1, 1)]
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=True
        )
        
        if time.time() < deadline:
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
    except Exception:
        pass
        
    # If time remains, do a local polish
    if time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs

# EVOLVE-BLOCK-END
