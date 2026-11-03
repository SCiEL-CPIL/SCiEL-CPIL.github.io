import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def _compute_objective(coeffs):
    a0, a4, a8 = coeffs
    # Calculate a12 such that P(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Numerical integration for norms
    # Using a fixed grid for speed and consistency
    x = np.linspace(-8, 8, 1000)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
    
    exp_2pi = np.exp(-2 * np.pi * x**2)
    exp_4pi = np.exp(-4 * np.pi * x**2)
    
    norm2_sq = np.sum(P**2 * exp_2pi) * dx
    norm4_sq = np.sum(P**4 * exp_4pi) * dx
    
    norm2_sq_hat = np.sum(P_minus**2 * exp_2pi) * dx
    norm4_sq_hat = np.sum(P_minus**4 * exp_4pi) * dx
    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_objective(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Multi-start local optimization
    # We use L-BFGS-B which handles bounds and is efficient for smooth functions
    starts = [
        x0,
        x0 + np.array([0.1, 0.01, 0.0001]),
        x0 - np.array([0.1, 0.01, 0.0001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.5, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002])
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_compute_objective, s, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END
