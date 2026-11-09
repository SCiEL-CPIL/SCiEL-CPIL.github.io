"""Combined program: Seed initialization with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 600.0 - 30.0  # Safety margin
    
    # The seed program (Inspiration 1) scores 1.0, which is the best known.
    # The current program (Laguerre roots) scores 0.48, which is worse.
    # We start from the seed configuration and attempt to refine it.
    
    # Initial guess: Arithmetic progression from the seed
    x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Objective function: 
    # We approximate the C4 upper bound minimization by minimizing the 
    # sup-norm of the monic polynomial formed by the roots.
    # This is a standard technique in polynomial approximation to minimize oscillations.
    def objective(roots):
        # Ensure roots are sorted and positive
        roots = np.sort(roots)
        if np.any(roots <= 0):
            return 1e10
        
        # Build the polynomial P(x) = prod (x - r_i)
        # We evaluate on a grid to find the sup-norm
        max_root = np.max(roots)
        # Grid from 0 to 1.2 * max_root
        grid = np.linspace(0, max_root * 1.2, 200)
        
        # Evaluate P(x) efficiently using log-sum-exp to avoid overflow
        # log|P(x)| = sum log|x - r_i|
        # We want to minimize max |P(x)|
        # To avoid numerical issues, we can normalize by the leading coefficient (which is 1)
        # and just compute the max absolute value.
        
        # For 12 roots, direct multiplication is fine, but values can be large.
        # Let's compute the max log-absolute value.
        log_abs_vals = np.zeros_like(grid)
        for r in roots:
            # Add log|x - r|
            # Handle x == r (log 0 = -inf). But roots are discrete, grid might hit them.
            # If grid hits a root, P(x)=0, log is -inf. This is good (minimizes max).
            # We use a small epsilon to avoid -inf in sum if we were summing, 
            # but here we sum logs.
            # Actually, if x=r, the term is 0. 
            # We can just compute the values directly if we scale.
            pass
            
        # Direct computation with scaling
        # P(x) can be huge. Let's compute the max of |P(x)| / (scale)
        # Scale by the product of distances from a reference point?
        # Simpler: Just compute the values and take the max. 
        # 12 roots, x ~ 50. (50)^12 ~ 2.4e20. This fits in float64.
        
        P_vals = np.ones_like(grid)
        for r in roots:
            P_vals *= (grid - r)
            
        sup_norm = np.max(np.abs(P_vals))
        
        # We want to minimize this sup norm.
        # However, the sup norm depends on the scale of the roots.
        # If we shift all roots to the right, the sup norm increases.
        # If we compress roots, the sup norm might decrease.
        # This objective might push roots to 0 or compress them.
        # We need to ensure the "uncertainty" is minimized, which might imply a specific spread.
        
        # Alternative Objective: Minimize the variance of the roots?
        # Or minimize the last root?
        # Given the ambiguity, and that the seed is already good, 
        # a local search with a weak gradient might just stay near the seed.
        
        return sup_norm

    # Use Nelder-Mead for local refinement
    # We restrict the search to positive values by transforming the variables?
    # Or just use bounds? Nelder-Mead doesn't support bounds well.
    # We'll use a penalty in the objective if roots <= 0.
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    try:
        # Run optimization
        res = minimize(objective, best_x, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass

    # Ensure final roots are positive and sorted
    best_x = np.sort(best_x)
    best_x = np.maximum(best_x, 1e-6)
    
    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END
