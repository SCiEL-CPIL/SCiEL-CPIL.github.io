"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search focused on the guaranteed minimizer region.
    
    Strategy:
    1. The problem states the minimizer lies in [-4, 4]^5.
    2. We use a scrambled Sobol sequence (deterministic via fixed seed) for space-filling properties.
    3. We map the points to [-4, 4]^5 to maximize density in the region of interest.
    4. We evaluate the first 500 points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        from scipy.stats import qmc
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Stage 1: Global Exploration
        # Use 250 points for global search
        n_global = 250
        sobol_global = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit_global = sobol_global.random(n_global)
        # Map to [-4, 4]
        points_global = points_unit_global * 8.0 - 4.0
        
        for i in range(n_global):
            if calls_made >= budget:
                break
            x = points_global[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Stage 2: Local Refinement
        # Use remaining budget for local search around the best point found
        n_local = budget - calls_made
        if n_local > 0 and best_x is not None:
            # Generate local Sobol points
            # We want to search in a box around best_x
            # Size of local box: 4.0 (i.e., +/- 2.0) seems reasonable for Schwefel-like functions
            local_width = 4.0
            
            sobol_local = qmc.Sobol(d=dim, scramble=True, seed=43)
            # Generate enough points, take the first n_local
            # Note: n_local might not be a power of 2, but Sobol generator handles arbitrary N
            points_unit_local = sobol_local.random(n_local)
            
            # Map local points: center at best_x, spread by local_width
            # points_unit_local is in [0, 1]. Center is 0.5.
            # Offset = (points_unit_local - 0.5) * local_width
            # local_x = best_x + offset
            points_local = best_x + (points_unit_local - 0.5) * local_width
            
            # Clip to valid box [-4, 4]
            points_local = np.clip(points_local, -4.0, 4.0)
            
            for i in range(n_local):
                x = points_local[i]
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
