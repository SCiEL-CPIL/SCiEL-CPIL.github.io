"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f near x0 by modeling it as a global optimization task.
    We minimize f(x)^2 using scipy.optimize.minimize_scalar with expanding bounds.
    This avoids manual bracketing and Newton's method overhead.
    """
    x0 = float(x0)
    
    # Helper to compute objective safely
    def objective(x):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return 1e20
            return fx * fx
        except Exception:
            return 1e20

    # Check if x0 is already a root
    try:
        fx0 = f(x0)
        if math.isfinite(fx0) and abs(fx0) <= 1e-8:
            return x0
    except Exception:
        pass

    best_x = x0
    best_val = objective(x0)
    
    # Strategy: Try minimization with expanding bounds centered at x0.
    # Polynomials are smooth, so the root is likely in a reasonable neighborhood.
    # We start small and expand to ensure we don't miss the root if it's further away.
    # Each minimize_scalar call uses ~10-20 function evaluations.
    # Total budget is 500, so we can afford ~10-20 minimization attempts.
    
    scales = [1.0, 10.0, 100.0, 1000.0, 1e6]
    
    for scale in scales:
        a = x0 - scale
        b = x0 + scale
        
        # Skip if bounds are not valid (e.g., x0 is huge and scale is small, but float handles this)
        # Ensure a < b
        if not (a < b):
            continue
            
        try:
            result = minimize_scalar(objective, bounds=(a, b), method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
            
            if result.success and math.isfinite(result.fun):
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
                    
                    # If we found a very good root, we can stop early
                    if best_val <= 1e-16: # |f(x)| <= 1e-8
                        return best_x
                        
        except Exception:
            continue

    # Final check
    if math.isfinite(best_x):
        try:
            if abs(f(best_x)) <= 1e-8:
                return best_x
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
