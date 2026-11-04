"""Refined uncertainty inequality solution using generalized Laguerre roots."""
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Optimizes the parameter alpha of the generalized Laguerre polynomial L_12^(alpha)
    to minimize the expected C4 upper bound, proxied by the sum of the roots.
    """
    n_roots = 12
    
    # We hypothesize that minimizing the "scale" of the roots minimizes the C4 bound.
    # The sum of roots for L_n^(alpha) is n(n + alpha + 1) ... wait, sum is 2n-1 + n*alpha.
    # Let's verify: For n=12, sum = 23 + 12*alpha.
    # Minimizing this requires minimizing alpha.
    # Constraint: alpha > -1 for positive roots.
    
    # We will search for alpha in a range close to -1 to minimize the roots.
    # However, we should check if there's an optimum.
    # Let's evaluate the sum of squares as a more robust "size" metric.
    
    best_roots = None
    best_metric = float('inf')
    
    # Search alphas from -0.95 to 2.0
    # Use a fine grid near -1
    alphas = np.concatenate([
        np.linspace(-0.95, -0.5, 20),
        np.linspace(-0.5, 2.0, 30)
    ])
    
    for alpha in alphas:
        try:
            nodes, weights = roots_genlaguerre(n_roots, alpha)
            # Check validity
            if not np.all(nodes > 0):
                continue
            
            # Metric: Sum of squares (proxy for energy/variance)
            metric = np.sum(nodes**2)
            
            if metric < best_metric:
                best_metric = metric
                best_roots = nodes
        except Exception:
            continue
            
    if best_roots is None:
        # Fallback
        nodes, _ = roots_genlaguerre(n_roots, 0.0)
        return list(nodes)
        
    return list(best_roots)
