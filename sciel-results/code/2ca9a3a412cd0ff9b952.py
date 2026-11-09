"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import genlaguerre
from numpy.polynomial.laguerre import laggauss

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_bound(roots, nodes, weights):
    """
    Compute the uncertainty bound (proxy for C4) using Gauss-Laguerre quadrature.
    The bound is related to the integral of the squared logarithmic derivative
    of the double-root polynomial.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) == 0:
        return 1e10
    
    # Check validity
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 1e-6):
        return 1e10

    # Calculate logarithmic derivative squared: (sum 2/(x - x_i))^2
    # To avoid numerical instability from singularities, we rely on the fact that
    # the Gauss-Laguerre nodes are fixed and we just evaluate the function there.
    # The integral is finite.
    
    # Vectorized calculation:
    # diff shape: (N_nodes, M_roots)
    # We compute 2 / (nodes[:, None] - roots[None, :])
    
    # To prevent division by zero if a node hits a root (unlikely but possible),
    # we add a tiny epsilon or rely on the fact that GL nodes don't hit roots usually.
    # However, if they do, the term is singular. The integral is still finite.
    # A more robust way is to sum the terms.
    
    try:
        # x is nodes (N,), roots is roots (M,)
        # term = 2 / (x_i - x_j)
        # We sum over j for each i, then square, then weight sum.
        
        # Use broadcasting
        # nodes: (N, 1), roots: (1, M)
        diff = nodes[:, None] - roots[None, :]
        
        # Avoid division by zero
        # If diff is very small, the term is large.
        # We can use np.where to handle exact zeros, but float precision usually avoids it.
        log_deriv_sq = np.sum(2.0 / diff, axis=1) ** 2
        
        # Integrate: sum w_i * f(x_i)
        bound_val = np.sum(weights * log_deriv_sq)
        
        if not np.isfinite(bound_val):
            return 1e10
            
        return bound_val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    
    # Parameters
    m = 12
    n = 1
    
    # Precompute Gauss-Laguerre nodes and weights
    # We need enough nodes to resolve the peaks around the 12 roots.
    # 200 nodes should be sufficient for m=12.
    num_nodes = 200
    try:
        nodes, weights = laggauss(num_nodes)
    except Exception:
        # Fallback if laggauss fails
        nodes, weights = laggauss(100)
    
    # Initial guess: Zeros of L_m^{(1)}(x)
    # These are positive and distinct.
    try:
        L_init = genlaguerre(m, 1)
        init_roots = L_init.roots()
        init_roots = np.real(init_roots)
        init_roots = np.sort(init_roots)
        
        # Ensure they are positive
        init_roots = init_roots[init_roots > 0]
        
        if len(init_roots) != m:
            # If we didn't get exactly m positive roots, use a heuristic spacing
            init_roots = np.array([1.0 + i * 2.0 for i in range(m)])
    except Exception:
        init_roots = np.array([1.0 + i * 2.0 for i in range(m)])
        
    best_roots = init_roots.copy()
    best_score = _compute_bound(best_roots, nodes, weights)
    
    # Optimization
    # We want to minimize the bound.
    # Constraints: roots > 0 and sorted.
    # L-BFGS-B works with bounds. We can enforce sorting by optimizing in log-space
    # or just using bounds and checking validity in the objective.
    # Better: Optimize in log-space to ensure positivity.
    # Let y_i = log(x_i). Then x_i = exp(y_i).
    # Sorting constraint: x_i < x_{i+1} => y_i < y_{i+1}.
    # This is still a constraint.
    # Alternative: Just use L-BFGS-B with bounds [1e-6, None] and rely on the
    # objective penalty for unsorted roots? No, L-BFGS-B doesn't handle inequality
    # constraints between variables directly.
    
    # Strategy: Use SLSQP or Trust-Constr with constraints?
    # Or, simpler: Use L-BFGS-B on the original variables with wide bounds,
    # and if the result is unsorted, sort it and re-evaluate.
    # However, the optimizer might converge to an unsorted local minimum.
    
    # Let's try optimizing the roots directly with L-BFGS-B, but we will
    # sort them after each step? No, that breaks gradients.
    
    # Let's use the fact that the objective is symmetric with respect to permutation
    # of roots? No, the integral is not symmetric if we don't sum over all permutations?
    # Wait, the integral $\int (\sum \frac{1}{x-x_i})^2$ IS symmetric under permutation of roots.
    # So the optimizer doesn't care about the order, but the "roots" must be distinct.
    # The Hessian is positive definite in the space of distinct roots.
    # So we can just optimize with bounds [1e-6, 1000].
    
    bounds = [(1e-6, 1000.0)] * m
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed + _SAFETY_MARGIN > _DEADLINE:
            return 1e10
        return _compute_bound(params, nodes, weights)
    
    try:
        res = minimize(
            objective,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_score:
            best_roots = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Polish with a different method if time permits
    if not _deadline_reached():
        # Sort the roots to ensure a valid sequence for return, though the score is permutation invariant
        best_roots = np.sort(best_roots)
        
        try:
            res2 = minimize(
                objective,
                best_roots,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res2.fun < best_score:
                best_roots = res2.x
                best_score = res2.fun
        except Exception:
            pass
    
    # Final validation and sorting
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    # Ensure strict increasing order for the "double root" structure validity
    # If two roots are too close, push them apart slightly
    for i in range(len(best_roots) - 1):
        if best_roots[i+1] - best_roots[i] < 1e-5:
            best_roots[i+1:] += 1e-5
    
    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
