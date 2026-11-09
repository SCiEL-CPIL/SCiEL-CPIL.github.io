"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 180.0  # Safety margin
    
    # Hermite polynomial values at 0
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 1039500
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Coefficients of Hermite polynomials for x^2 moments (n + 1/2)
    # <x^2> for psi_n is n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Weights (squared coefficients)
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15 # Avoid division by zero, penalize 0 norm
            
        # Variance (effective n)
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        # Assuming isotropic symmetry, var_xi = var_x
        # Product of variances
        return (var_x * var_x)

    # Seed coefficients
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_coeffs = seed_coeffs
    best_score = objective(best_coeffs)
    
    # Analytical candidate: Minimize weight on higher n
    # Theoretical minimum is achieved by c4=0, c8=0
    candidate_analytical = np.array([1.0, 0.0, 0.0])
    score_analytical = objective(candidate_analytical)
    
    if score_analytical < best_score:
        best_score = score_analytical
        best_coeffs = candidate_analytical

    # Try to optimize using Nelder-Mead
    try:
        from scipy.optimize import minimize
        
        # Use multiple restarts if time permits
        max_restarts = 5
        restarts_done = 0
        
        for i in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Perturb the seed slightly for the first few restarts
            if i == 0:
                x0 = seed_coeffs
            else:
                # Random perturbation
                noise = np.random.normal(0, 0.1, size=3)
                x0 = seed_coeffs + noise
            
            result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-12})
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
                restarts_done += 1
                
            # Early exit if we hit the theoretical minimum for Gaussian (0.25)
            if best_score <= 0.2500001:
                break
                
    except Exception:
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
