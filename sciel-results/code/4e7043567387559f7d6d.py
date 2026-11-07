"""Deterministic multi-start coordinate-wise descent for Katsuura-like f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic hierarchical coordinate-wise descent.
    
    Uses a sequence of 1D grid scans along each coordinate axis,
    starting with a coarse global grid and refining locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return float(func(x))

        # Initial point: Center
        current_x = np.zeros(dim)
        current_f = evaluate(current_x)

        if calls >= budget:
            return current_x, current_f

        # Define refinement phases
        # (radius, num_points_per_dim)
        # We ensure num_points is odd so the center is included, 
        # though we don't strictly need it if we check bounds.
        phases = [
            (5.0, 11),   # Coarse global scan
            (1.0, 21),   # Medium scan
            (0.25, 21),  # Fine scan
            (0.05, 21),  # Very fine scan
            (0.01, 21),  # Ultra fine scan
            (0.002, 13)  # Final polish
        ]

        for radius, num_points in phases:
            if calls >= budget:
                break
            
            # Generate grid points for one dimension
            # We use linspace from -radius to +radius
            # Clip to [-5, 5] is handled by func, but we should avoid
            # generating points way outside if possible to save "logic" 
            # though func clips.
            grid_1d = np.linspace(-radius, radius, num_points)
            
            for d in range(dim):
                if calls >= budget:
                    break
                
                for val in grid_1d:
                    if calls >= budget:
                        break
                    
                    # Create candidate
                    candidate = current_x.copy()
                    # The value is relative to current_x[d]? 
                    # No, the grid is absolute values relative to the center of the search space?
                    # The previous logic in the prompt's inspiration suggests 
                    # "fixed grid of candidates per dimension".
                    # If we are doing coordinate descent, we usually scan 
                    # around the current best.
                    # So candidate[d] = current_x[d] + val? 
                    # Or candidate[d] = val?
                    # If radius is 5.0, and current_x is 0, val is [-5, 5].
                    # If current_x is 2.0, and we want to search around it, 
                    # we should search [2.0 - 5.0, 2.0 + 5.0].
                    # However, if we just use absolute values from a global grid, 
                    # it's not coordinate descent around the current best.
                    
                    # Let's use relative offsets:
                    # candidate[d] = current_x[d] + val
                    # This makes it a local search around the current best.
                    # But for the first phase, current_x is 0, so it searches [-5, 5].
                    # This is correct.
                    
                    candidate[d] = current_x[d] + val
                    candidate[d] = np.clip(candidate[d], -5.0, 5.0)
                    
                    f_val = evaluate(candidate)
                    if f_val < current_f:
                        current_f = f_val
                        current_x = candidate

        return current_x, current_f
# EVOLVE-BLOCK-END
