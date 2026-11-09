import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 600.0
    m = 12
    
    def compute_c4_bound(roots):
        """Compute the actual C4 upper bound from root locations.
        
        For a polynomial P(x) = prod_i (x - r_i)^2 with double roots,
        the C4 bound involves the integral:
        C4 = integral_0^inf |P(x)|^2 * e^{-x} / (x^2 + 1) dx
        
        We compute this numerically using Gauss-Laguerre quadrature.
        """
        if len(roots) < 2:
            return 1e10
        
        # Use Gauss-Laguerre quadrature with enough nodes
        n_quad = max(200, 4 * m)
        x_nodes, x_weights = roots_laguerre(n_quad)
        
        # Compute P(x) = prod_i (x - r_i)^2 at quadrature nodes
        # To avoid overflow/underflow, compute in log space
        log_abs_P = np.zeros(n_quad)
        for r in roots:
            log_abs_P += 2.0 * np.log(np.abs(x_nodes - r) + 1e-300)
        
        abs_P = np.exp(np.clip(log_abs_P, -700, 700))
        
        # Integrate |P(x)|^2 * e^{-x} / (x^2 + 1)
        integrand = abs_P**2 / (x_nodes**2 + 1.0)
        c4 = np.sum(x_weights * integrand)
        
        return c4
    
    def compute_score(roots):
        """Wrapper for compute_c4_bound with safety checks."""
        try:
            if any(r <= 0 for r in roots):
                return 1e10
            if any(np.isnan(r) or np.isinf(r) for r in roots):
                return 1e10
            return compute_c4_bound(roots)
        except Exception:
            return 1e10
    
    def optimize_roots(initial_roots):
        """Optimize root positions to minimize the C4 bound."""
        def objective(params):
            roots = params
            # Penalty for non-positive roots
            if np.any(roots <= 0):
                return 1e10
            # Penalty for very close roots (numerical instability)
            min_dist = min(abs(roots[i] - roots[j]) for i in range(m) for j in range(i+1, m))
            if min_dist < 1e-6:
                return 1e10
            return compute_c4_bound(roots)
        
        # Use L-BFGS-B with bounds
        bounds = [(1e-4, None)] * m
        result = minimize(objective, initial_roots, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        return result.x, result.fun
    
    # Generate diverse initial configurations
    best_roots = None
    best_score = 1e10
    
    # Strategy 1: Start from Laguerre roots with various alpha
    for alpha in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        if time.time() - start_time > deadline - 60:
            break
        
        try:
            roots_std, _ = roots_laguerre(m, alpha)
            roots_std = sorted(roots_std)
            
            # Add small perturbation to break symmetry
            np.random.seed(int(alpha * 100))
            perturbed = roots_std + np.random.uniform(-0.1, 0.1, m)
            perturbed = np.maximum(perturbed, 1e-3)
            
            opt_roots, opt_score = optimize_roots(perturbed)
            
            if opt_score < best_score and np.all(opt_roots > 0):
                best_score = opt_score
                best_roots = opt_roots.tolist()
                
        except Exception:
            continue
    
    # Strategy 2: Start from geometric sequences
    for ratio in [1.2, 1.5, 2.0, 2.5]:
        if time.time() - start_time > deadline - 60:
            break
        
        try:
            initial = np.array([ratio**i for i in range(m)])
            opt_roots, opt_score = optimize_roots(initial)
            
            if opt_score < best_score and np.all(opt_roots > 0):
                best_score = opt_score
                best_roots = opt_roots.tolist()
                
        except Exception:
            continue
    
    # Strategy 3: Start from uniform grid
    for spacing in [1.0, 1.5, 2.0, 3.0]:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            initial = np.array([spacing * (i + 1) for i in range(m)])
            opt_roots, opt_score = optimize_roots(initial)
            
            if opt_score < best_score and np.all(opt_roots > 0):
                best_score = opt_score
                best_roots = opt_roots.tolist()
                
        except Exception:
            continue
    
    # If no good result found, use a reasonable default
    if best_roots is None or len(best_roots) != m:
        best_roots = [4.0 * (i + 1) for i in range(m)]
    
    # Final check
    if time.time() - start_time > deadline - 10:
        return [4.0 * (i + 1) for i in range(m)]
    
    return list(best_roots)
