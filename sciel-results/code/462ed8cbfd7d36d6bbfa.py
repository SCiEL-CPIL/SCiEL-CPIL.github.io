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

        # Phase 1: Coarse grid search to identify promising regions
        # Evaluate corners and center of the box
        n_corners = 2 ** dim
        coarse_points = []
        for i in range(n_corners):
            point = np.zeros(dim)
            for d in range(dim):
                bit = (i >> d) & 1
                point[d] = -5.0 + bit * 10.0
            coarse_points.append(point)
        coarse_points.append(np.zeros(dim))
        
        for x in coarse_points:
            if time.time() > deadline - 50 or calls_made >= budget - 50:
                break
            counted_func(x)

        # Phase 2: Adaptive grid refinement around best region
        # Start with the region around the best point found
        # For Rastrigin, local minima are spaced about 2 units apart
        # So we use a region of size ~4 units around best_x
        
        local_region_size = 4.0
        n_local = 2 ** dim
        
        local_points = []
        for i in range(n_local):
            point = best_x.copy()
            for d in range(dim):
                bit = (i >> d) & 1
                if bit == 0:
                    point[d] -= local_region_size / 2
                else:
                    point[d] += local_region_size / 2
            local_points.append(point)
        
        # Evaluate local grid
        for x in local_points:
            if time.time() > deadline - 50 or calls_made >= budget - 50:
                break
            x = np.clip(x, -5, 5)
            counted_func(x)

        # Phase 3: Multi-start Powell with adaptive restarts
        # Use remaining budget for local optimization
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
                    x0 = best_x + rng.normal(0, 1.0, dim)
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
