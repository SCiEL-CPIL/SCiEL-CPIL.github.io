import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-resolution grid search for global coverage
        # Rastrigin has periodic structure with local minima spaced ~2 units apart
        # Use nested grids of increasing density to rapidly identify the correct basin
        
        # Level 1: Coarse grid (3 points per dim = 3^5 = 243 points)
        # This gives spacing of 5 units in [-5, 5]
        grid_levels = [3, 5, 7]
        
        for n_per_dim in grid_levels:
            if time.time() > deadline - 100 or calls_made >= budget - 100:
                break
            
            # Generate grid points
            coords = np.linspace(-5, 5, n_per_dim)
            grid_points = np.array(np.meshgrid(*[coords]*dim)).T.reshape(-1, dim)
            
            # Evaluate grid points
            for x in grid_points:
                if time.time() > deadline - 100 or calls_made >= budget - 100:
                    break
                counted_func(x)
        
        # Phase 2: Local refinement around best point
        # Use a fine grid in a local region around best_x
        local_radius = 1.5
        n_local_per_dim = 5
        
        local_coords = np.linspace(best_x - local_radius, best_x + local_radius, n_local_per_dim)
        local_coords = np.clip(local_coords, -5, 5)
        
        local_grid = np.array(np.meshgrid(*[local_coords]*dim)).T.reshape(-1, dim)
        local_grid = np.clip(local_grid, -5, 5)
        
        for x in local_grid:
            if time.time() > deadline - 50 or calls_made >= budget - 50:
                break
            counted_func(x)
        
        # Phase 3: Multi-start Powell with adaptive restarts
        remaining_budget = budget - calls_made
        if remaining_budget > 20:
            n_restarts = min(5, max(1, remaining_budget // 50))
            evals_per_restart = remaining_budget // n_restarts
            
            for i in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_made >= budget:
                    break
                
                # Start from best point or random point
                if i == 0:
                    x0 = best_x.copy()
                else:
                    x0 = best_x + rng.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)
                
                res = minimize(counted_func, x0, method='Powell', 
                              options={'maxfev': evals_per_restart, 
                                       'xtol': 1e-8, 'ftol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

        # Phase 4: Polish with Nelder-Mead
        if calls_made < budget - 10 and time.time() < deadline - 5:
            remaining = budget - calls_made
            res = minimize(counted_func, best_x, method='Nelder-Mead',
                          options={'maxfev': remaining, 
                                   'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        return best_x, best_f
