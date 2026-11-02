"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time


def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # We need to integrate polynomials of degree up to 26 times exp(-x^2)
    # Hermite quadrature with n nodes is exact for degree up to 2n-1
    # So n=14 is sufficient for degree 26, use n=30 for safety
    n_quad = 30
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    # Hermite polynomials (physicists' convention) evaluated at nodes
    def H0_vals(x):
        return np.ones_like(x, dtype=float)
    
    def H4_vals(x):
        return x**4 - 6*x**2 + 3
    
    def H8_vals(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12_vals(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly_vals(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0_vals(x) + c4*H4_vals(x) + c8*H8_vals(x) + c12*H12_vals(x)
    
    def poly_deriv_vals(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
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
        
        try:
            P = poly_vals(x_nodes, c0, c4, c8)
            Pp = poly_deriv_vals(x_nodes, c0, c4, c8)
            
            # I0 = ∫P²e^(-x²)dx
            I0 = np.sum(w_nodes * P**2)
            if I0 <= 0 or not np.isfinite(I0):
                return np.inf
            
            # Ix2 = ∫x²P²e^(-x²)dx
            Ix2 = np.sum(w_nodes * x_nodes**2 * P**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
            
            # Ideriv = ∫(P'-xP)²e^(-x²)dx
            Ideriv = np.sum(w_nodes * (Pp - x_nodes*P)**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    # 3D grid search for better initial points - now much faster with Gauss-Hermite
    if time.time() < deadline - 30:
        # Coarse grid search
        c0_range = np.linspace(-2.0, 0.5, 20)
        c4_range = np.linspace(-0.5, 0.5, 20)
        c8_range = np.linspace(-0.1, 0.1, 15)
        
        best_grid_score = np.inf
        best_grid_coeffs = x0.copy()
        
        for c0 in c0_range:
            if time.time() > deadline - 20:
                break
            for c4 in c4_range:
                if time.time() > deadline - 15:
                    break
                for c8 in c8_range:
                    if time.time() > deadline - 10:
                        break
                    coeffs = np.array([c0, c4, c8])
                    score = compute_uncertainty_product(coeffs)
                    if score < best_grid_score and np.isfinite(score):
                        best_grid_score = score
                        best_grid_coeffs = coeffs.copy()
        
        if best_grid_score < best_score:
            best_score = best_grid_score
            best_coeffs = best_grid_coeffs.copy()
    
    # Nelder-Mead optimization from multiple starting points
    starting_points = [x0, best_coeffs]
    
    # Add diverse starting points
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        noise = np.random.normal(0, 0.3, 3)
        starting_points.append(best_coeffs + noise)
    
    for sp in starting_points:
        if time.time() > deadline - 15:
            break
        try:
            res = opt.minimize(compute_uncertainty_product, sp, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Local refinement around the best found
    if time.time() < deadline - 10:
        # Fine local search with multiple restarts
        np.random.seed(123)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            noise_scale = 0.03 / (1 + _ * 0.15)
            noise = np.random.normal(0, noise_scale, 3)
            x_try = best_coeffs + noise
            try:
                res = opt.minimize(compute_uncertainty_product, x_try, method='Nelder-Mead', 
                                  options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 500, 'adaptive': True})
                if res.fun < best_score and np.isfinite(res.fun):
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish using L-BFGS-B if available
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, method='L-BFGS-B', 
                              options={'gtol': 1e-14, 'maxiter': 1000})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Return the coefficients c0, c4, c8
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
