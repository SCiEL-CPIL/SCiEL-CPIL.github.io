import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite
import time

def _compute_objective(coeffs):
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    # a0 - 12a4 + 1680a8 - 332640 a12 = 0
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Define P(x) and P_minus(x)
    # P(x) = a0 H0(x) + a4 H4(x) + a8 H8(x) + a12 H12(x)
    # P_minus(x) = a0 H0(x) + a4 H4(x) + a8 H8(x) - a12 H12(x)
    
    def integrand_pos_2(x):
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        return (P**2) * np.exp(-2 * np.pi * x**2)

    def integrand_pos_4(x):
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        return (P**4) * np.exp(-4 * np.pi * x**2)

    def integrand_hat_2(x):
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
        return (P_minus**2) * np.exp(-2 * np.pi * x**2)

    def integrand_hat_4(x):
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
        return (P_minus**4) * np.exp(-4 * np.pi * x**2)

    # Integration limits: [-10, 10] is safe for exp(-2*pi*x^2)
    # quad is adaptive, so it will focus where the function is non-zero
    limit = 10
    
    try:
        norm2_sq, _ = quad(integrand_pos_2, -limit, limit, limit=100)
        norm4_sq, _ = quad(integrand_pos_4, -limit, limit, limit=100)
        norm2_sq_hat, _ = quad(integrand_hat_2, -limit, limit, limit=100)
        norm4_sq_hat, _ = quad(integrand_hat_4, -limit, limit, limit=100)
    except Exception:
        return 1e10

    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = x0
    best_score = _compute_objective(x0)
    
    # Multi-start local optimization
    # We use L-BFGS-B which handles bounds and is efficient for smooth functions
    # We generate starts around the seed and some random perturbations
    starts = [
        x0,
        x0 + np.array([0.1, 0.01, 0.001]),
        x0 - np.array([0.1, 0.01, 0.001]),
        x0 + np.array([0.5, 0.05, 0.005]),
        x0 - np.array([0.5, 0.05, 0.005]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.5, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.13, 0.04, 0.0003]), # Slight perturbation of seed
        np.array([-1.13, 0.04, 0.0004]),
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_compute_objective, s, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # If we have time left, do a final polish with Nelder-Mead or higher precision
    if time.time() < deadline - 30:
        try:
            # Nelder-Mead is derivative-free and can sometimes escape local minima
            res_nm = minimize(_compute_objective, best_result, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
        except Exception:
            pass
            
    return best_result
