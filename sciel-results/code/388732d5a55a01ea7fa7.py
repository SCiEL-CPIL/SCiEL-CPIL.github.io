import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s kill
    
    # Known state-of-the-art coefficients (Goncalves et al. 2017)
    c0_known = -113.0 / 100.0
    c4_known = 1.0 / 25.0
    c8_known = 1.0 / 3240.0
    
    best_coeffs = np.array([c0_known, c4_known, c8_known])
    best_score = float('inf')
    
    # Helper to calculate c12 from c0, c4, c8 such that g(0) = 0
    # H_n(0) values for even n (physicist's Hermite polynomials):
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -10395
    def get_c12(c0, c4, c8):
        return (c0 - 3.0 * c4 + 105.0 * c8) / 10395.0
    
    def compute_uncertainty_product(c0, c4, c8):
        """Compute Δx · Δk for g(x) = P(x) exp(-x²/2) with g(0)=0."""
        c12 = get_c12(c0, c4, c8)
        
        def g(x):
            H0 = eval_hermite(0, x)
            H4 = eval_hermite(4, x)
            H8 = eval_hermite(8, x)
            H12 = eval_hermite(12, x)
            P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
            return P * np.exp(-x**2 / 2.0)
        
        def gp(x):
            # d/dx [H_n(x) exp(-x²/2)] = 2n H_{n-1}(x) exp(-x²/2)
            # H_0 term: 0
            # H_4 term: 8 H_3(x) exp(-x²/2)
            # H_8 term: 16 H_7(x) exp(-x²/2)
            # H_12 term: 24 H_11(x) exp(-x²/2)
            H3 = eval_hermite(3, x)
            H7 = eval_hermite(7, x)
            H11 = eval_hermite(11, x)
            return (8.0 * c4 * H3 + 16.0 * c8 * H7 + 24.0 * c12 * H11) * np.exp(-x**2 / 2.0)
        
        try:
            I_g2, _ = quad(lambda x: g(x)**2, -np.inf, np.inf, limit=200)
            I_x2g2, _ = quad(lambda x: x**2 * g(x)**2, -np.inf, np.inf, limit=200)
            I_gp2, _ = quad(lambda x: gp(x)**2, -np.inf, np.inf, limit=200)
        except Exception:
            return 1e100
        
        if I_g2 < 1e-15:
            return 1e100
        
        # By symmetry, <x> = 0
        delta_x2 = I_x2g2 / I_g2
        delta_k2 = I_gp2 / I_g2
        
        if delta_x2 < 0 or delta_k2 < 0:
            return 1e100
        
        return np.sqrt(delta_x2 * delta_k2)
    
    # Evaluate baseline
    baseline_val = compute_uncertainty_product(c0_known, c4_known, c8_known)
    best_score = baseline_val
    
    if time.time() - start_time > deadline - 30:
        return best_coeffs
    
    # Build quadratic form for the uncertainty product squared
    # We want to minimize (Δx * Δk)^2 = (I_x2g2 * I_gp2) / I_g2^2
    # This is a ratio of quadratic forms in the coefficients
    # Let's use a grid search in a transformed space
    
    # First, let's evaluate the Hessian of the objective function numerically
    # to understand the curvature and transform the space
    
    # Sample points to build a quadratic approximation
    def objective_squared(coeffs):
        val = compute_uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
        return val * val
    
    # Build a grid of points around the known optimum
    center = best_coeffs
    scale = 0.5
    
    # Generate grid points for Hessian estimation
    h = 0.01
    e1 = np.array([h, 0, 0])
    e2 = np.array([0, h, 0])
    e3 = np.array([0, 0, h])
    
    # Central differences for Hessian
    f0 = objective_squared(center)
    
    def hessian_component(i, j):
        if i == j:
            fi = objective_squared(center + e1 * (1 if i == 0 else 0) + e2 * (1 if i == 1 else 0) + e3 * (1 if i == 2 else 0))
            fm = objective_squared(center - e1 * (1 if i == 0 else 0) - e2 * (1 if i == 1 else 0) - e3 * (1 if i == 2 else 0))
            return (fi - 2*f0 + fm) / (h*h)
        else:
            ei = e1 * (1 if i == 0 else 0) + e2 * (1 if i == 1 else 0) + e3 * (1 if i == 2 else 0)
            ej = e1 * (1 if j == 0 else 0) + e2 * (1 if j == 1 else 0) + e3 * (1 if j == 2 else 0)
            fp = objective_squared(center + ei + ej)
            fm = objective_squared(center - ei - ej)
            fp1 = objective_squared(center + ei - ej)
            fm1 = objective_squared(center - ei + ej)
            return (fp - fp1 - fm + fm1) / (4*h*h)
    
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            H[i, j] = hessian_component(i, j)
            H[j, i] = H[i, j]
    
    # Cholesky decomposition to transform to isotropic space
    try:
        L = np.linalg.cholesky(H + 1e-10 * np.eye(3))  # Add small regularization
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        L_inv = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Grid search in transformed space
    grid_size = 15
    search_radius = 3.0  # In transformed space
    
    best_val = best_score
    best_transformed = np.zeros(3)
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if time.time() - start_time > deadline - 10:
                    break
                tx = -search_radius + 2 * search_radius * i / (grid_size - 1)
                ty = -search_radius + 2 * search_radius * j / (grid_size - 1)
                tz = -search_radius + 2 * search_radius * k / (grid_size - 1)
                
                t = np.array([tx, ty, tz])
                coeffs = center + L_inv @ t
                
                val = compute_uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
                if val < best_val:
                    best_val = val
                    best_transformed = t.copy()
                    best_coeffs = coeffs.copy()
    
    # Refine around the best point found
    if time.time() - start_time < deadline - 5:
        # Local refinement using Nelder-Mead from the best grid point
        try:
            def objective_refine(coeffs):
                if time.time() - start_time > deadline - 5:
                    return best_score
                return compute_uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
            
            result = minimize(
                objective_refine,
                best_coeffs,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
    
    if time.time() - start_time > deadline - 5:
        return best_coeffs
    
    return best_coeffs
