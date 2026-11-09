import numpy as np
import time
import sys
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using
    CMA-ES (via scipy's dual-annealing or similar global search if CMA-ES not available,
    but here we implement a robust global search strategy using scipy.optimize.dual_annealing
    or a custom CMA-ES loop if available. Since standard scipy doesn't always have CMA-ES 
    in older versions, we will use a robust global optimizer like 'dual_annealing' which 
    combines basin hopping with simulated annealing, or 'shgo' (Simplicial Homology Global Optimizer).
    
    However, the prompt suggests CMA-ES. scipy.optimize does not have CMA-ES built-in in all versions.
    We will implement a simple CMA-ES from scratch or use 'dual_annealing' as a robust global alternative.
    Given the constraint of 600s, 'dual_annealing' is often very effective for global optimization 
    in moderate dimensions (21D is small).
    
    Actually, let's check if we can use 'dual_annealing' or 'shgo'. 
    'dual_annealing' is generally better for smooth functions.
    
    Strategy:
    1. Precompute the basis functions Y_m on the grid.
    2. Define the objective function R(c) for real coefficients c (size 21).
    3. Use scipy.optimize.dual_annealing to minimize R(c).
    4. Return the optimal c.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    cos_theta_vals = x_gl
    n_grid = 70 * n_phi
    
    # 2. Pre-compute Real Basis Functions Y_m
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m (m>0) = sqrt(2) N_m P_10^m(cos theta) cos(m phi)
    # Y_m (m<0) = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi)
    
    Y_real = np.zeros((n_coeffs, n_grid))
    
    for m_idx in range(n_coeffs):
        m = m_idx - k # m ranges from -10 to 10
        
        if m == 0:
            num = factorial(k)
            den = factorial(k)
            N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
            P_vals = lpmv(0, k, cos_theta_vals)
            # Y_0 is independent of phi
            Y_vals = np.tile(N_m * P_vals[:, np.newaxis], (1, n_phi))
            Y_real[m_idx, :] = Y_vals.flatten()
        else:
            abs_m = abs(m)
            num = factorial(k - abs_m)
            den = factorial(k + abs_m)
            N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
            
            P_vals = lpmv(abs_m, k, cos_theta_vals)
            
            if m > 0:
                trig_term = np.cos(m * phi)
            else:
                trig_term = np.sin(abs_m * phi)
            
            # Y_m = sqrt(2) N_m P^m trig_term
            Y_vals = (np.sqrt(2.0) * N_m * P_vals[:, np.newaxis] * trig_term[np.newaxis, :])
            Y_real[m_idx, :] = Y_vals.flatten()
    
    # Quadrature weights
    # The integral is sum_{i,j} w_i dphi f(x_i, phi_j)^p
    weights = np.tile(w_gl, (n_phi,)) * dphi
    weights = weights.flatten()
    
    def compute_R(c):
        """
        Computes R = ||f||_4 / ||f||_2 for coefficients c.
        """
        if not np.all(np.isfinite(c)):
            return np.inf
        f = Y_real @ c
        
        norm2_sq = np.sum(weights * f**2)
        if norm2_sq < 1e-15:
            return np.inf
            
        norm2 = np.sqrt(norm2_sq)
        norm4_four = np.sum(weights * f**4)
        norm4 = np.sqrt(np.sqrt(norm4_four))
        
        return norm4 / norm2
    
    # Optimization
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    # Initial guess: Random
    np.random.seed(42)
    x0 = np.random.randn(n_coeffs)
    
    # Bounds for c to keep them in a reasonable range for the optimizer
    # The problem allows max|c| up to 1e5, but R is scale invariant.
    # We can fix the scale by normalizing c in the objective or just let the optimizer find a direction.
    # Since R is scale invariant, we should probably normalize c in the objective to help the optimizer.
    # However, dual_annealing works well with unnormalized variables if the function is flat in radial direction.
    # To be safe, let's normalize c in the objective function to a unit sphere.
    
    def objective_normalized(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            return np.inf
        c_norm = c / norm_c
        return compute_R(c_norm)
    
    # We will use dual_annealing which is a global optimizer
    # It requires a function and optionally bounds.
    # Since we normalize inside, bounds are less critical, but let's set them to [-10, 10] to avoid huge numbers.
    bounds = [(-10.0, 10.0)] * n_coeffs
    
    best_c = None
    best_R = np.inf
    
    # Run dual_annealing
    # maxiter and other parameters can be tuned.
    # With 21 dimensions, it should converge reasonably fast.
    try:
        res = minimize(
            objective_normalized, 
            x0, 
            method='dual_annealing',
            bounds=bounds,
            options={
                'maxiter': 200, 
                'restarts': 10, 
                'popsize': 50, 
                'maxiter_no_improvement': 100,
                'verbose': 0
            }
        )
        
        if res.fun < best_R:
            best_R = res.fun
            best_c = res.x
    except Exception as e:
        print(f"Error in dual_annealing: {e}")
        
    # If dual_annealing didn't find a good solution or timed out, try a few local optimizations from random starts
    # to ensure we have a valid result.
    if best_c is None or best_R > 1.1: # Threshold for "good enough"
        for i in range(5):
            if time.time() - start_time > time_limit - safety_margin:
                break
            x0_local = np.random.randn(n_coeffs)
            try:
                res_local = minimize(
                    objective_normalized, 
                    x0_local, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res_local.fun < best_R:
                    best_R = res_local.fun
                    best_c = res_local.x
            except Exception:
                pass
    
    if best_c is None:
        best_c = x0 # Fallback
        
    # Normalize best_c to ensure validity bounds
    # max|c| should be between 1e-3 and 1e5
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c *= (1e-2 / max_c)
    elif max_c > 1e5:
        best_c *= (1e4 / max_c)
        
    return best_c

# EVOLVE-BLOCK-END
