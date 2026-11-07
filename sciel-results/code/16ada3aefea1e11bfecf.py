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
        deadline = start_time + 280.0  # Safety margin before 300s kill

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to identify promising regions
        # Use a 2^dim grid over the entire box
        n_coarse = 2 ** dim  # 32 points for dim=5
        
        # Generate coarse grid points
        coarse_points = []
        for i in range(n_coarse):
            point = np.zeros(dim)
            for d in range(dim):
                bit = (i >> d) & 1
                point[d] = -5.0 + bit * 10.0  # -5 or 5
            coarse_points.append(point)
        
        # Add the center point
        coarse_points.append(np.zeros(dim))
        
        # Evaluate coarse grid
        for x in coarse_points:
            if time.time() > deadline - 50 or calls_made >= budget - 50:
                break
            counted_func(x)
        
        # Phase 2: Adaptive grid refinement around best region
        # Start with the region around the best point found
        # Use a local grid of size 2^dim around best_x
        
        # Determine a reasonable local region size
        # For Rastrigin, the local minima are spaced about 2 units apart
        # So we use a region of size ~4 units around best_x
        local_region_size = 4.0
        
        # Create a local grid around best_x
        # Divide the region into 2^dim sub-cells
        n_local = 2 ** dim
        
        local_points = []
        for i in range(n_local):
            point = np.zeros(dim)
            for d in range(dim):
                bit = (i >> d) & 1
                # Offset from center by half the region size
                offset = -local_region_size / 2 + bit * local_region_size
                point[d] = best_x[d] + offset
            point = np.clip(point, -5, 5)
            local_points.append(point)
        
        # Add the center (best_x)
        local_points.append(best_x.copy())
        
        # Evaluate local grid
        for x in local_points:
            if time.time() > deadline - 30 or calls_made >= budget - 30:
                break
            counted_func(x)
        
        # Phase 3: Multi-restart Powell for local refinement
        # Use the best point found so far as the primary start
        # And a few random restarts from the coarse grid
        
        n_restarts = 4
        evals_per_restart = min(60, max(20, (budget - calls_made) // n_restarts))
        
        # Start points: best_x, and a few from the coarse grid
        start_points = [best_x.copy()]
        for x in coarse_points[:3]:
            if x is not best_x:
                start_points.append(x.copy())
        
        for i, x0 in enumerate(start_points):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget - 10:
                break
            
            # Run Powell
            res = minimize(counted_func, x0, method='Powell', 
                         options={'maxfev': evals_per_restart, 
                                  'xtol': 1e-8, 'ftol': 1e-8})
            
            # Update best if better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 4: Polish with Nelder-Mead if budget remains
        if calls_made < budget - 10 and time.time() < deadline - 5:
            remaining = budget - calls_made
            res = minimize(counted_func, best_x, method='Nelder-Mead', 
                         options={'maxfev': remaining, 
                                  'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 5: Final random perturbation search if budget remains
        if calls_made < budget - 3 and time.time() < deadline - 3:
            rng = np.random.default_rng(123)
            n_extra = min(budget - calls_made, 20)
            for _ in range(n_extra):
                if calls_made >= budget or time.time() > deadline - 2:
                    break
                x_try = best_x + rng.normal(0, 0.1, dim)
                x_try = np.clip(x_try, -5, 5)
                counted_func(x_try)
        
        return best_x, best_f
