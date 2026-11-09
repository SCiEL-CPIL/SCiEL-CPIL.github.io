import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize, differential_evolution
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600  # seconds
    
    def laguerre_norm_sq(roots):
        """Compute squared Laguerre norm of P(x) = prod (x - r_i)^2"""
        # Build polynomial as product of (x - r_i)^2 for each root
        # Start with polynomial [1] (constant 1)
        poly = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2r*x + r^2
            factor = np.array([r*r, -2*r, 1.0])
            poly = np.convolve(poly, factor)
        
        # Convert from standard basis to Laguerre basis
        # Use laguerre.poly2lag
        lag_coeffs = laguerre.poly2lag(poly)
        
        # The squared norm in Laguerre inner product:
        # For Laguerre polynomials, <L_m, L_n> = delta_mn
        # So ||P||^2 = sum(c_k^2) where P = sum c_k L_k
        norm_sq = np.sum(lag_coeffs**2)
        return norm_sq
    
    def cost_func(s_vars):
        """Cost function: s_vars are sqrt of roots, so roots = s_vars^2"""
        roots = s_vars**2
        try:
            return laguerre_norm_sq(roots)
        except Exception:
            return 1e10
    
    # Generate diverse starting points from Laguerre-Gauss nodes
    # Use different numbers of nodes and perturbations
    start_points = []
    
    # Primary: Laguerre-Gauss nodes for n=12
    nodes_12, _ = laguerre.laggauss(12)
    start_points.append(np.sqrt(nodes_12))
    
    # Perturbations of the primary
    rng = np.random.RandomState(42)
    for i in range(8):
        noise = rng.normal(0, 0.1, size=12)
        perturbed = nodes_12 * (1 + noise)
        perturbed = np.maximum(perturbed, 0.01)  # Ensure positive
        start_points.append(np.sqrt(perturbed))
    
    # Different Laguerre-Gauss node counts (projected to 12D)
    for n_nodes in [10, 14, 16]:
        nodes_n, _ = laguerre.laggauss(n_nodes)
        if n_nodes >= 12:
            # Take first 12 or random subset
            idx = rng.choice(n_nodes, size=12, replace=False)
            start_points.append(np.sqrt(nodes_n[idx]))
        else:
            # Pad with interpolated values
            x_interp = np.linspace(nodes_n[0], nodes_n[-1], 12)
            y_interp = np.interp(x_interp, nodes_n, nodes_n)
            start_points.append(np.sqrt(np.maximum(y_interp, 0.01)))
    
    # Local optimization with L-BFGS-B from each starting point
    best_val = float('inf')
    best_s = start_points[0].copy()
    
    for i, s_init in enumerate(start_points):
        if time.time() - start_time > deadline - 60:
            break
        
        try:
            result = minimize(cost_func, s_init, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_s = result.x.copy()
        except Exception:
            pass
    
    # Check time
    if time.time() - start_time > deadline - 30:
        return [float(x**2) for x in best_s]
    
    # Basin hopping from the best solution found
    def basin_hopping_step(s):
        """Perform one basin hopping step: perturb then polish"""
        # Perturb in sqrt-space
        noise = np.random.normal(0, 0.2, size=12)
        s_perturbed = s + noise
        s_perturbed = np.maximum(s_perturbed, 0.01)
        
        # Polish with L-BFGS-B
        try:
            result = minimize(cost_func, s_perturbed, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success:
                return result.x, result.fun
        except Exception:
            pass
        
        return s_perturbed, cost_func(s_perturbed)
    
    # Run basin hopping iterations
    current_s = best_s.copy()
    current_val = best_val
    rng_bh = np.random.RandomState(123)
    
    for iteration in range(50):
        if time.time() - start_time > deadline - 30:
            break
        
        new_s, new_val = basin_hopping_step(current_s)
        
        # Accept if better or with Metropolis criterion (simplified: always accept if better)
        if new_val < current_val:
            current_s = new_s
            current_val = new_val
        
        if current_val < best_val:
            best_val = current_val
            best_s = current_s.copy()
    
    # Final local polish
    try:
        result = minimize(cost_func, best_s, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_s = result.x
    except Exception:
        pass
    
    # Return the roots (squared values of s_vars)
    roots = [float(x**2) for x in best_s]
    
    # Verify all positive
    for r in roots:
        if r <= 0:
            roots = [float(x**2) for x in start_points[0]]
            break
    
    return roots
# EVOLVE-BLOCK-END
