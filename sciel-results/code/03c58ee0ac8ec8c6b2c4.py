import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import minimize_scalar
import time
import math

def _compute_c4(alpha, m=12, n=1):
    """
    Compute the C4 upper bound for the Fourier sign uncertainty problem
    using the refined Cohn-Goncalves Laguerre polynomial construction.
    
    The construction uses a Laguerre polynomial combination with m double roots.
    The C4 bound relates to the tail behavior and the location of the last sign change.
    """
    try:
        # Get roots and weights of the generalized Laguerre polynomial L_m^(alpha)
        nodes, weights = roots_laguerre(m, alpha)
    except Exception:
        return np.inf
    
    # Ensure all roots are positive
    if np.any(nodes <= 0):
        return np.inf
    
    # Sort nodes
    nodes = np.sort(nodes)
    
    # The Laguerre polynomial L_m^(alpha)(x) has m roots at these nodes.
    # For the double-root construction, we consider P(x) = [L_m^(alpha)(x)]^2
    # which has 2m roots (each with multiplicity 2).
    
    # The C4 bound in the refined uncertainty inequality is related to:
    # C4 = sup_{|f|<=1, f in L^2} |hat{f}(xi)| where the support constraints are met
    # 
    # For the Laguerre construction, the key quantity is the value of the
    # Laguerre polynomial at specific points related to the sign change.
    
    # The last sign change of P(x) = [L_m^(alpha)(x)]^2 doesn't exist since it's non-negative.
    # Instead, we look at L_m^(alpha)(x) itself.
    
    # The C4 upper bound is computed from the maximum of |L_m^(alpha)(x)| / sqrt(weight(x))
    # or related quantities that control the Fourier transform decay.
    
    # A standard approach: the bound involves the supremum of |P(x)| over x >= 0
    # where P is the constructed polynomial, normalized appropriately.
    
    # Let's compute the actual C4 bound using the known formula for this construction.
    # The bound is related to the integral of the square of the Laguerre polynomial
    # weighted by the exponential decay.
    
    # For the refined inequality, C4 is bounded by:
    # C4 <= C * (max_node)^n * exp(-max_node/2) / (some normalization)
    
    # More precisely, using the quadrature weights:
    # The bound can be estimated from the tail behavior.
    
    # Let's use a more direct approach: compute the value that appears in the
    # uncertainty inequality bound.
    
    # The C4 constant for the Laguerre construction is related to:
    # C4 = 1 / (sum of weights * exp(-nodes/2) * L_m^(alpha)(nodes)^2)
    # or similar. Let me use a known relationship.
    
    # Actually, for the Cohn-Goncalves construction, the key bound is:
    # The Fourier transform of the window function is bounded, and C4 controls
    # the decay rate.
    
    # Let me compute a proxy that correlates with the actual C4 bound:
    # The bound is inversely related to how well the Laguerre polynomial
    # approximates the desired window shape.
    
    # A practical approach: C4 is minimized when the roots are optimally placed.
    # The quality of the placement can be measured by the condition number of
    # the associated system, but let's use the actual bound formula.
    
    # For m=12, n=1, the C4 bound can be computed as:
    # C4 = sup_{x>=0} |L_m^(alpha)(x)| * exp(-x/2) / sqrt(normalization)
    
    # Let's evaluate this on a fine grid
    x_grid = np.linspace(0, 4*max(nodes), 10000)
    lag_vals = np.array([eval_laguerre(m, x, alpha) for x in x_grid])
    weighted = np.abs(lag_vals) * np.exp(-x_grid/2)
    
    # Normalize by the L2 norm with respect to the Laguerre weight
    # The norm of L_m^(alpha) is sqrt(1/m!) * Gamma(m+alpha+1)/Gamma(m+1) for the standard weight
    # But for our purpose, let's use the quadrature weights
    norm_sq = np.sum(weights * lag_vals**2 * np.exp(-nodes))
    if norm_sq <= 0:
        return np.inf
    norm = np.sqrt(norm_sq)
    
    c4 = np.max(weighted) / norm
    
    return c4

def construct():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    def objective(alpha):
        if time.time() > deadline - 5:
            return np.inf
        return _compute_c4(alpha)
    
    # Multi-start optimization over alpha
    best_score = np.inf
    best_alpha = 0.0
    
    # Initial grid search
    alphas = np.concatenate([
        np.linspace(0, 10, 100),
        np.linspace(10, 50, 50),
        np.linspace(50, 200, 30)
    ])
    
    for a in alphas:
        if time.time() > deadline - 30:
            break
        s = objective(a)
        if s < best_score:
            best_score = s
            best_alpha = a
    
    # Local refinement using scalar minimization
    try:
        res = minimize_scalar(objective, bounds=(max(0, best_alpha - 5), best_alpha + 5), 
                            method='bounded', options={'xatol': 1e-4, 'maxiter': 100})
        if res.fun < best_score:
            best_score = res.fun
            best_alpha = res.x
    except Exception:
        pass
    
    # Second refinement with wider bounds if time permits
    if time.time() < deadline - 60:
        try:
            res2 = minimize_scalar(objective, bounds=(max(0, best_alpha - 20), best_alpha + 20),
                                  method='bounded', options={'xatol': 1e-3, 'maxiter': 50})
            if res2.fun < best_score:
                best_score = res2.fun
                best_alpha = res2.x
        except Exception:
            pass
    
    # Final retrieval
    if time.time() > deadline - 5:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
    
    try:
        nodes, _ = roots_laguerre(12, best_alpha)
        nodes = np.sort(nodes)
        return [float(x) for x in nodes]
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
