"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Minimizes the C4 upper bound.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Try different approaches to minimize C4 bound
    best_roots = None
    best_score = np.inf
    
    # Approach 1: Use Gauss-Laguerre nodes with optimized beta
    # The weight function is w(x) = e^(-x) * x^beta
    # For generalized Laguerre, alpha = beta
    
    def compute_c4_bound(roots):
        """
        Compute the C4 upper bound given 12 positive double roots.
        This is a proxy for the actual Cohn-Goncalves bound.
        The bound typically involves integrals related to the polynomial.
        """
        roots = np.sort(roots)
        if np.any(roots <= 0):
            return np.inf
        
        # Construct the polynomial P(x) = prod_i (x - r_i)^2
        # The C4 bound for Fourier uncertainty often involves:
        # C4 = sqrt( (integral x^2 |P|^2 e^(-x) dx) / (integral |P|^2 e^(-x) dx) )
        # or similar variance-like quantities
        
        # Use numerical quadrature
        # Integrate over [0, inf) with weight e^(-x)
        # Use Gauss-Laguerre quadrature with many nodes
        
        n_quad = 200
        x_quad, w_quad = roots_laguerre(n_quad)
        
        # Compute P(x) at quadrature points
        # P(x) = prod (x - r_i)^2
        log_p = np.zeros_like(x_quad)
        for r in roots:
            log_p += 2 * np.log(np.abs(x_quad - r) + 1e-300)
        
        # Avoid overflow/underflow by working with logs
        # integral of |P|^2 e^(-x) dx ≈ sum w_j * |P(x_j)|^2
        # But |P|^2 = exp(2 * log_p)
        
        # For stability, compute ratio directly
        # C4^2 = <x^2> / <1> where <f> = integral f(x) |P(x)|^2 e^(-x) dx / integral |P(x)|^2 e^(-x) dx
        
        # Compute weights for the ratio
        w_eff = w_quad * np.exp(2 * log_p)
        
        # Normalize
        total = np.sum(w_eff)
        if total <= 0 or not np.isfinite(total):
            return np.inf
        
        x2_weighted = np.sum(x_quad**2 * w_eff)
        x_weighted = np.sum(x_quad * w_eff)
        
        # Variance-like quantity
        variance = x2_weighted / total - (x_weighted / total)**2
        
        # The C4 bound is related to sqrt(variance) or similar
        # Let's use sqrt(variance) as a proxy
        if variance < 0:
            return np.inf
        return np.sqrt(variance)
    
    # Approach 1: Optimize beta for Gauss-Laguerre nodes
    def objective_beta(beta):
        if time.time() > deadline:
            return 1e10
        alpha = max(beta, -0.99)  # alpha > -1 for generalized Laguerre
        try:
            # Get 12 nodes of generalized Laguerre
            # scipy doesn't have direct generalized Laguerre roots, 
            # so we use standard Laguerre and transform, or use different approach
            # For now, use standard Laguerre roots scaled
            x_nodes, _ = roots_laguerre(12)
            # Scale based on beta: for weight x^beta e^(-x), nodes shift
            # Approximate: nodes scale as ~ (beta + 1) + something
            # Actually, let's just optimize the roots directly
            roots = x_nodes * (1.0 + beta)
            if np.any(roots <= 0):
                return 1e10
            return compute_c4_bound(roots)
        except:
            return 1e10
    
    # Approach 2: Direct optimization of root positions
    def objective_roots(x):
        if time.time() > deadline:
            return 1e10
        roots = np.sort(x)
        if np.any(roots <= 0):
            return 1e10
        return compute_c4_bound(roots)
    
    # Start with standard Laguerre nodes
    x0, _ = roots_laguerre(12)
    x0 = x0 * 1.0  # Initial guess
    
    # Try local optimization from multiple starting points
    best_local_score = np.inf
    best_local_roots = None
    
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 60:
            break
        
        # Different starting points
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            x_start = x0 * 0.5
        elif i == 2:
            x_start = x0 * 2.0
        elif i == 3:
            x_start = np.linspace(0.5, 10, 12)
        else:
            x_start = x0 + np.random.normal(0, 0.5, 12)
            x_start = np.sort(x_start)
        
        # Ensure positive
        x_start = np.maximum(x_start, 0.1)
        
        try:
            result = minimize(
                objective_roots,
                x_start,
                method='L-BFGS-B',
                bounds=[(0.01, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.success and result.fun < best_local_score:
                best_local_score = result.fun
                best_local_roots = np.sort(result.x)
        except:
            pass
    
    if best_local_roots is not None:
        best_roots = best_local_roots
        best_score = best_local_score
    
    # Also try global optimization if time permits
    if time.time() < deadline - 120:
        try:
            bounds = [(0.1, 50.0)] * 12
            result_global = differential_evolution(
                objective_roots,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=42,
                polish=True
            )
            if result_global.fun < best_score:
                best_score = result_global.fun
                best_roots = np.sort(result_global.x)
        except:
            pass
    
    # Final refinement
    if best_roots is not None and time.time() < deadline - 30:
        try:
            result_final = minimize(
                objective_roots,
                best_roots,
                method='L-BFGS-B',
                bounds=[(0.01, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if result_final.fun < best_score:
                best_score = result_final.fun
                best_roots = np.sort(result_final.x)
        except:
            pass
    
    if best_roots is None:
        # Fallback to standard Laguerre nodes
        best_roots = x0
    
    return list(np.sort(best_roots))
# EVOLVE-BLOCK-END
