"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
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
            return 1e15  # Avoid division by zero, penalize 0 norm
            
        # Variance (effective n)
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        # Product of variances (isotropic assumption)
        return (var_x * var_x)

    # Seed coefficients from the original program
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Analytical candidate: pure H0 (c4=0, c8=0) gives c12 ≈ 0, var_x ≈ 0.5
    candidate_analytical = np.array([1.0, 0.0, 0.0])
    
    # Evaluate both starting points
    best_coeffs = seed_coeffs.copy()
    best_score = objective(best_coeffs)
    
    score_analytical = objective(candidate_analytical)
    if score_analytical < best_score:
        best_score = score_analytical
        best_coeffs = candidate_analytical.copy()
    
    # Try to optimize using Nelder-Mead
    try:
        from scipy.optimize import minimize
        
        max_restarts = 8
        restarts_done = 0
        
        for i in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            # Different starting points for diversity
            if i == 0:
                x0 = best_coeffs.copy()
            elif i == 1:
                x0 = seed_coeffs.copy()
            elif i == 2:
                x0 = candidate_analytical.copy()
            elif i == 3:
                # Small perturbation around analytical
                x0 = candidate_analytical + np.array([0.0, 0.01, 0.001])
            elif i == 4:
                # Different sign pattern
                x0 = np.array([1.0, -0.01, 0.001])
            elif i == 5:
                # Perturb seed
                x0 = seed_coeffs + np.random.normal(0, 0.05, size=3)
            elif i == 6:
                # Another analytical variant
                x0 = np.array([0.5, 0.0, 0.0])
            else:
                # Random restart
                x0 = np.random.normal(0, 0.5, size=3)
            
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
                restarts_done += 1
                
            # Early exit if we hit the theoretical minimum for Gaussian (0.25)
            if best_score <= 0.25000001:
                break
        
        # Final high-precision polish pass with tighter tolerances
        if time.time() < deadline - 5:
            polish_result = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_coeffs = polish_result.x.copy()
                
    except Exception:
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
