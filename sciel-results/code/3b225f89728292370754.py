"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware optimizer using grid search and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Grid Search
        # The global minimum of Rastrigin is at integer coordinates in the transformed space.
        # With shift and rotation, the minimum in the original space is likely near
        # integer coordinates or simple fractions.
        # We sample a grid of points to identify the basin of the global minimum.
        
        # Grid values: -4, 0, 4 covers the range [-4, 4] with spacing 4.
        # This is 3^5 = 243 points.
        grid_vals = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        grid_points = [np.array(p) for p in itertools.product(grid_vals, repeat=dim)]
        
        grid_results = []
        for pt in grid_points:
            if call_count >= budget or time.time() > deadline:
                break
            # Clip to box just in case, though grid is within [-5,5]
            pt_clipped = np.clip(pt, -5, 5)
            fx = call_func(pt_clipped)
            if fx != np.inf:
                grid_results.append((fx, pt_clipped))
        
        if not grid_results:
            return best_x, best_f

        # Sort by function value
        grid_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement
        # Run Nelder-Mead on the top 3 points from the grid search.
        remaining = budget - call_count
        if remaining > 0:
            # Determine how many refinements we can afford
            # Each Nelder-Mead run needs some minimum calls to be effective
            min_calls_per_refine = 30
            n_refine = min(3, remaining // min_calls_per_refine)
            
            if n_refine > 0:
                calls_per_refine = remaining // n_refine
                
                for i in range(n_refine):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    fx, pt = grid_results[i]
                    
                    try:
                        res = minimize(
                            call_func, 
                            pt, 
                            method='Nelder-Mead',
                            options={'maxfev': calls_per_refine, 'xatol': 1e-8, 'fatol': 1e-8}
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
