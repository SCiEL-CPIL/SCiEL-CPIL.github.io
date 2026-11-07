import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Global Grid + Nelder-Mead Optimizer.
    
    Strategy:
    1. Perform a coarse global search using a grid (or sparse sampling) to find the approximate basin of attraction.
       For dim=5, a full grid is too expensive (11^5 = 161k points > 500 budget). 
       Instead, we use a sparse initial set of points or a coarse grid with fewer points per dimension if possible, 
       or simply a set of random points + a few structured points.
       Actually, for dim=5 and budget=500, a grid of 3 points per dimension is 3^5 = 243 points. This fits!
       This allows us to robustly find the region where the minimum likely lies, even with rotation/shift.
       
    2. Once the best point from the grid is found, use Nelder-Mead simplex method to refine.
       Nelder-Mead is derivative-free and very effective for smooth, low-dimensional problems like Rosenbrock.
       It adapts the simplex shape to the local geometry, similar to CMA-ES but often faster for smooth functions
       when started near the basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Global Coarse Search
        # For dim=5, we can use a grid of 3 points per dimension: 3^5 = 243 evaluations.
        # This leaves 500 - 243 = 257 evaluations for local refinement.
        # The points are -4, 0, 4. This covers the likely range of the minimum [-4, 4]^5.
        
        grid_points_per_dim = 3
        total_grid_points = grid_points_per_dim ** dim
        
        if total_grid_points <= budget // 2:
            # Perform grid search
            # Create grid coordinates
            coords = np.linspace(-4, 4, grid_points_per_dim)
            
            # Generate all combinations
            grid = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)
            
            # Evaluate grid points
            for x in grid:
                if calls_made >= budget:
                    break
                tracked_func(x)
            
            # If we ran out of budget during grid, we can't do local refinement.
            # But 243 < 500, so we should be fine.
            
        else:
            # Fallback to random sampling if dim is higher (not the case here, but for robustness)
            n_random = budget // 2
            rng = np.random.default_rng(42)
            random_points = rng.uniform(-4, 4, size=(n_random, dim))
            for x in random_points:
                if calls_made >= budget:
                    break
                tracked_func(x)

        if best_x is None:
            # Should not happen if budget > 0
            return np.zeros(dim), np.inf

        # Phase 2: Local Refinement with Nelder-Mead
        # We have remaining budget. Let's use it for Nelder-Mead.
        # Nelder-Mead can be run with a specific maxiter or until convergence.
        # We need to be careful about the number of function calls Nelder-Mead makes.
        # Each iteration of Nelder-Mead in dim D typically makes 1 new function evaluation.
        # So we can allow up to (budget - calls_made) iterations.
        
        remaining_budget = budget - calls_made
        
        # Nelder-Mead options
        # maxiter: maximum number of iterations. Each iteration is roughly 1 function call.
        # tol: tolerance for convergence.
        
        # We want to use the remaining budget efficiently.
        # Let's set maxiter to remaining_budget.
        
        try:
            result = minimize(
                tracked_func, 
                best_x, 
                method='Nelder-Mead',
                options={
                    'maxiter': remaining_budget,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            # Update best_x and best_f from result if it's better
            # Note: tracked_func already updates best_x and best_f internally.
            # But result.x is the final point. Let's ensure we have the best one.
            # The tracked_func keeps the global best.
            
        except Exception:
            # If Nelder-Mead fails, we still have the best from the grid phase.
            pass

        # Ensure we return the best found
        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = tracked_func(best_x)

        return best_x, best_f
