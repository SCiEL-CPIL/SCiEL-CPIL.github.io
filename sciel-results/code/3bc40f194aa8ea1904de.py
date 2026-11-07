"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Sobol grid exploration followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin for 300s limit
        
        # State tracking
        best_x = np.full(self.dim, np.nan)
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        # Wrapped function to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return 1e10  # Penalize to stop optimization
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Strategy:
        # 1. Evaluate a Sobol quasi-random grid of 5^5 = 3125 points? 
        #    No, budget is 500. 
        #    Let's use a smaller grid. 3^5 = 243 points is a good baseline for coverage.
        #    Or maybe 4^5 = 1024 is too many. 
        #    Let's stick with 3^5 = 243 points for the grid phase.
        #    This leaves ~257 calls for refinement.
        
        # Generate grid points: values at -4, 0, 4
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*([grid_vals]*self.dim))).T.reshape(-1, self.dim)
        
        # Evaluate grid
        grid_fvals = np.empty(len(grid_points))
        for i, p in enumerate(grid_points):
            if calls_made >= max_calls:
                break
            grid_fvals[i] = wrapped_func(p)
        
        # If we ran out of calls during grid, return best found
        if calls_made >= max_calls:
            return best_x, best_f

        # Find top K points for refinement
        # Sort by function value
        sorted_indices = np.argsort(grid_fvals)
        top_k = 5
        top_starts = grid_points[sorted_indices[:top_k]]
        
        # Refine using Nelder-Mead
        # Nelder-Mead is derivative-free and robust.
        # We need to be careful with the number of function calls.
        # Each Nelder-Mead step takes dim+1 = 6 calls.
        # We have ~257 calls left. 257 / 6 ~ 42 steps per run.
        # 5 runs * 42 steps = 210 steps. Total calls ~ 3125? No, 243 + 5*42*6 = 243 + 1260 = 1503. Too many.
        # We need to limit maxiter.
        # Let's allocate remaining calls evenly among top_k starts.
        remaining_calls = max_calls - calls_made
        calls_per_run = max(10, remaining_calls // top_k)
        max_iter = calls_per_run // (self.dim + 1)
        
        for start_point in top_starts:
            if calls_made >= max_calls:
                break
            if time.time() > deadline - 5.0:
                break
                
            try:
                # Use a small initial simplex to stay local
                # Default simplex in scipy is based on x0 + 0.05 * x0 (or 0.00025 if x0 is 0)
                # We want to control the size. 
                # Let's initialize the simplex manually or use options if possible.
                # scipy.optimize.minimize with method='Nelder-Mead' uses options:
                # 'initial_simplex', 'xatol', 'fatol', 'maxiter'
                
                # Create a small simplex around start_point
                # Step size 0.5 seems reasonable given the grid spacing of 4.
                step = 0.5
                simplex = np.zeros((self.dim + 1, self.dim))
                simplex[0] = start_point
                for i in range(self.dim):
                    simplex[i+1] = start_point
                    simplex[i+1][i] += step
                    
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    start_point, 
                    method='Nelder-Mead', 
                    options={
                        'initial_simplex': simplex,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'maxiter': max_iter
                    }
                )
                # wrapped_func already updates best_x and best_f
            except Exception:
                pass
            
            # If we are running low on calls, break
            if calls_made >= max_calls:
                break
                
        # Ensure best_x is not nan
        if np.isnan(best_x).any():
             # Fallback to first grid point
             best_x = grid_points[0]
             best_f = grid_fvals[0]
             
        return best_x, best_f
# EVOLVE-BLOCK-END
