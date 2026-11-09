"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial

def construct():
    start_time = time.time()
    deadline = 180.0  # seconds
    safety_margin = 5.0
    
    # Seed coefficients from the previous best
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    # Grid for numerical integration
    # The function is Gaussian-decaying, so a range of [-10, 10] is sufficient
    x = np.linspace(-10, 10, 2001)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2.0)
    
    # Precompute Hermite polynomials on the grid
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at x=0 for the constraint g(0)=0
    # H_n(0) = (-1)^(n/2) * (n-1)!! for even n
    # H0(0) = 1
    # H4(0) = 3!! = 3
    # H8(0) = 7!! = 105
    # H12(0) = 11!! = 10395
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def compute_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = compute_c12(c0, c4, c8)
        
        # Construct g(x)
        # Using standard Hermite polynomials multiplied by Gaussian
        # Note: Normalization constants cancel out in the ratio, 
        # but we include them for stability if we were checking norms.
        # Here we just need the shape.
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
        
        # Check if g is effectively zero (degenerate case)
        norm_sq = np.sum(g**2) * dx
        if norm_sq < 1e-10:
            return 1e100
            
        # Compute moments for uncertainty
        # <x>
        x_mean = np.sum(x * g**2) * dx / norm_sq
        # <x^2>
        x2_mean = np.sum(x**2 * g**2) * dx / norm_sq
        var_x = x2_mean - x_mean**2
        
        if var_x < 1e-10:
            return 1e100
            
        # Compute derivative g'(x)
        # g(x) = P(x) e^{-x^2/2}
        # g'(x) = (P'(x) - x P(x)) e^{-x^2/2}
        # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
        # P'(x) = c4 * 4 H3 + c8 * 8 H7 + c12 * 12 H11  (H0' = 0)
        
        H3 = eval_hermite(3, x)
        H7 = eval_hermite(7, x)
        H11 = eval_hermite(11, x)
        
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        P_prime = c4 * 4 * H3 + c8 * 8 * H7 + c12 * 12 * H11
        
        g_prime = (P_prime - x * P) * gauss
        
        # <p> in momentum space corresponds to i * Fourier shift, 
        # for real symmetric functions, <p> = 0. 
        # We calculate <p^2> = integral |g'|^2 / integral |g|^2
        p2_mean = np.sum(g_prime**2) * dx / norm_sq
        var_p = p2_mean # Assuming <p>=0 due to symmetry
        
        if var_p < 1e-10:
            return 1e100
            
        # Uncertainty product Delta x * Delta p
        delta_x = np.sqrt(var_x)
        delta_p = np.sqrt(var_p)
        
        return delta_x * delta_p

    best_coeffs = x0
    best_score = objective(x0)
    
    try:
        # Use L-BFGS-B for efficient gradient-based optimization
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success and result.fun < best_score:
            best_coeffs = result.x
            best_score = result.fun
        elif result.fun < best_score:
            best_coeffs = result.x
            best_score = result.fun
            
        # Check time
        if time.time() - start_time > deadline - safety_margin:
            return np.array(best_coeffs)
            
        # Try a local search from the best found so far with Nelder-Mead for refinement
        result2 = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if result2.fun < best_score:
            best_coeffs = result2.x
            best_score = result2.fun
            
    except Exception as e:
        # If optimization fails, return the seed
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
