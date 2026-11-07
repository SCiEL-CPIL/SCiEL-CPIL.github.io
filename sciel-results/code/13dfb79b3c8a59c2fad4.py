"""
Optimization strategy:
1. Grid-based coarse-to-fine deterministic scan: systematically partition
   the box into hypercubes, evaluate cell centers to identify low-value
   regions without relying on stochastic restarts.
2. Nelder-Mead local optimization from the best grid point for reliable
   convergence to the local minimum.
"""
# EVOLVE-BLOCK-START
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
        deadline = time.time() + 290.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Coarse-to-fine grid scan
        # Start with a coarse grid, then refine around the best region
        # For dim=5, we can afford a modest grid size
        
        # Phase 1a: Coarse grid - 3 points per dimension = 3^5 = 243 points
        # But we only have 500 budget total, so let's be more conservative
        # Use 2 points per dim first = 2^5 = 32 points, then refine
        
        # Actually, let's use a multi-resolution approach:
        # Level 0: 3^5 = 243 points (coarse)
        # If budget allows, Level 1: refine around best with 2^5 = 32 points in smaller region
        
        # Let's start with a reasonable coarse grid
        # With 500 budget, we can do:
        # - Coarse: 3^5 = 243 evals
        # - Refine: use remaining ~257 for Nelder-Mead or finer grid
        
        # Phase 1: Coarse grid scan
        # Use 3 points per dimension: -5, 0, 5
        grid_points = np.array(np.meshgrid(
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3)
        )).T.reshape(-1, dim)
        
        for point in grid_points:
            if check_deadline():
                break
            evaluate(point)
        
        # Phase 2: Refine around the best point found
        if best_x is not None and not check_deadline():
            # Create a finer grid around the best point
            # Use a box of radius 1.5 around best_x
            center = best_x.copy()
            radius = 1.5
            # 3 points per dim in the refined region
            refined_grid = np.array(np.meshgrid(
                np.linspace(center[0] - radius, center[0] + radius, 3),
                np.linspace(center[1] - radius, center[1] + radius, 3),
                np.linspace(center[2] - radius, center[2] + radius, 3),
                np.linspace(center[3] - radius, center[3] + radius, 3),
                np.linspace(center[4] - radius, center[4] + radius, 3)
            )).T.reshape(-1, dim)
            
            # Clip to bounds
            refined_grid = np.clip(refined_grid, -5, 5)
            
            for point in refined_grid:
                if check_deadline():
                    break
                evaluate(point)
        
        # Phase 3: Nelder-Mead local optimization from the best point
        if best_x is not None and not check_deadline():
            x0 = best_x.copy()
            
            # Use Nelder-Mead with bounded constraints
            bounds = [(-5, 5)] * dim
            
            # We need to track function calls for Nelder-Mead
            nm_calls = [0]
            
            def nm_func(x):
                nonlocal best_x, best_f
                nm_calls[0] += 1
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if result.x is not None:
                    x_nm = np.clip(result.x, -5, 5)
                    evaluate(x_nm)
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
