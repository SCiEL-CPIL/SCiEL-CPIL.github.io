"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based coarse-to-fine deterministic scan for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0

        def grid_scan_3d(center, size, func, dim, budget_remaining):
            """
            Evaluate a 3^dim grid over the hypercube centered at 'center' with 'size'.
            Returns best point, best value, and number of calls made.
            """
            # Generate 3 points per dimension
            # Points: center - size/2, center, center + size/2
            offsets = np.array([-size / 2.0, 0.0, size / 2.0])
            
            best_x = None
            best_f = float('inf')
            
            # Iterate over all combinations
            # Using itertools.product would be clean, but let's use nested loops or recursive generation
            # Since dim is small (5), we can use a simple recursive or product approach.
            # However, to avoid importing itertools, we can use a simple recursive function or np.meshgrid.
            
            # Using np.meshgrid for efficiency
            grids = np.meshgrid(*[center + offsets for _ in range(dim)], indexing='ij')
            points = np.stack([g.ravel() for g in grids], axis=1)
            
            # Check if we have enough budget
            num_points = points.shape[0]
            if num_points > budget_remaining:
                # If we don't have enough budget for the full grid, we should fall back to local search
                # Or just evaluate as many as we can? 
                # The contract says "More than 500 calls... makes the whole result invalid".
                # So we must not exceed budget.
                # If we can't do the full grid, we break out and use local search on the current center.
                return best_x, best_f, 0, False

            # Evaluate all points
            for x in points:
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
            
            return best_x, best_f, num_points, True

        center = np.zeros(dim)
        size = 10.0
        best_x = center.copy()
        best_f = float('inf')

        # Perform coarse-to-fine grid scans
        while True:
            grid_size = 3 ** dim
            if budget - calls_made < grid_size:
                break
            
            best_x, best_f, calls_used, success = grid_scan_3d(center, size, func, dim, budget - calls_made)
            if not success:
                break
                
            calls_made += calls_used
            center = best_x
            size /= 2.0

        # Use remaining budget for local refinement
        remaining = budget - calls_made
        if remaining > 0 and best_f < float('inf'):
            # Use Nelder-Mead or Powell
            # Powell is often better for multimodal functions if started in the right basin
            # Nelder-Mead is also good.
            try:
                # Limit maxiter to remaining budget to avoid exceeding
                # Note: minimize might call func more than maxiter in some implementations?
                # Usually maxiter is the max number of iterations, but each iteration might use multiple calls.
                # Nelder-Mead uses dim+1 calls per iteration roughly.
                # So maxiter should be conservative.
                max_iter = max(1, remaining // (dim + 1))
                
                result = minimize(func, center, method='Nelder-Mead', 
                                 options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_f:
                    best_x, best_f = result.x, result.fun
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
