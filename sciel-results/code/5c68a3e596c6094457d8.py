"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hybrid optimizer:
    1. Multi-start random sampling to find a promising basin.
    2. (1+1)-ES with adaptive step size for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        
        # Phase 1: Global Exploration
        # Use 20% of budget for multi-start sampling
        n_global = max(10, int(self.budget * 0.2))
        
        best_x = None
        best_f = np.inf
        
        # Sample n_global points uniformly
        for _ in range(n_global):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local Refinement
        # Use remaining budget for (1+1)-ES
        remaining = self.budget - n_global
        
        x = best_x.copy()
        fx = best_f
        sigma = 1.0 # Start with a moderate step size
        
        for _ in range(remaining):
            # Generate offspring
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Increase step size slightly on success
            else:
                sigma *= 0.9 # Decrease step size on failure
                
            # Ensure sigma doesn't get too small
            if sigma < 1e-6:
                sigma = 1e-6
                
            # Ensure sigma doesn't get too large (though clipping helps)
            if sigma > 10.0:
                sigma = 10.0

        return x, fx

# EVOLVE-BLOCK-END
