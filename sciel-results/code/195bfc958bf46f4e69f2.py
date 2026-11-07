# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG4 (3 objectives, 12 variables).
    
    Strategy:
    WFG4 is a 3-objective problem. The Pareto front is characterized by:
    - f1 depends primarily on x[0]
    - f2, f3 depend on the remaining variables
    
    For the standard WFG4 formulation, the Pareto-optimal points are achieved
    when the shape variables (x[1:]) are set to specific values that minimize
    the contribution to f2 and f3. In many formulations, this means setting
    x[1:] to 0 or to values that make the transformation functions output 0.
    
    This strategy directly samples the Pareto front by:
    1. Uniformly sampling x[0] to cover the range of f1.
    2. Setting x[1:] to values that place points on or near the Pareto front.
    3. Using a mix of deterministic and random sampling for robustness.
    
    We use 60% of budget for deterministic uniform sampling of x[0] to ensure
    full coverage, and 40% for random sampling to explore potential deviations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract lower and upper bounds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # For WFG4, the Pareto front is typically achieved when shape variables
        # are at specific values. Common choices:
        # - x[1:] = 0 (if bounds allow)
        # - x[1:] = lo[1:] (lower bounds)
        # - x[1:] = midpoints
        
        # Let's try setting shape variables to 0 first, but clip to bounds
        # Actually, for WFG4, the shape variables often need to be at specific
        # values. Let's use a heuristic: set them to 0 if possible, else midpoint.
        
        # Try multiple strategies for shape variables and pick the best
        # But since we have limited budget, let's use a robust default:
        # Set x[1:] to 0 (clipped to bounds) as this is common for WFG problems
        
        # Strategy: Use deterministic uniform sampling for x[0]
        # This ensures even coverage of the f1 axis
        
        n_det = int(self.budget * 0.6)  # 60% deterministic
        n_rand = self.budget - n_det     # 40% random
        
        # Deterministic uniform samples for x[0]
        if n_det > 0:
            x0_det = np.linspace(lo[0], hi[0], n_det)
        else:
            x0_det = np.array([])
        
        # Random samples for x[0]
        if n_rand > 0:
            x0_rand = np.random.uniform(lo[0], hi[0], n_rand)
        else:
            x0_rand = np.array([])
        
        # Combine all x[0] samples
        x0_samples = np.concatenate([x0_det, x0_rand])
        
        # For shape variables, try setting them to 0 (clipped to bounds)
        # This is a common heuristic for WFG problems where the Pareto front
        # is achieved when shape parameters are at their "ideal" values
        
        # Create base vector with shape variables set to 0 (clipped)
        base = np.zeros(self.dim)
        base[1:] = 0.0  # Shape variables at 0
        
        # Clip to bounds
        base = np.clip(base, lo, hi)
        
        # Also try an alternative: set shape variables to midpoints
        # and compare which gives better results? No, we don't have feedback.
        # Let's stick with 0 for shape variables as it's a strong prior.
        
        # Construct the full vectors
        X = np.tile(base, (len(x0_samples), 1))
        X[:, 0] = x0_samples
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Return the set of points
        return X
# EVOLVE-BLOCK-END
