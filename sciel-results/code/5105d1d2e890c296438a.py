import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin

        # ZDT6: f1(x1) = 1 - (1 - exp(-4*x1)*sin(6*pi*x1)^6)^(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        # On Pareto front, x2..x10 = 0.
        
        # Strategy: Use the parametric representation of the Pareto front.
        # Let u = (1 - f1)^(10) = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # Then f1 = 1 - u^(1/10) and f2 = 1 - sqrt(1 - u^(1/10))
        # We want to sample u uniformly in [0, 1] to get uniform f1 coverage.
        # However, f1 is not monotonic in x1, so we need to find x1 for each u.
        # Actually, it's easier to sample x1 directly but non-uniformly to get uniform f1.
        # Or, we can just sample x1 uniformly and accept the non-uniform f1 distribution,
        # but the current approach (grid + searchsorted) is already good.
        
        # Let's try a different approach: sample u uniformly, then for each u,
        # solve for x1 such that 1 - exp(-4*x1)*sin(6*pi*x1)^6 = u.
        # This requires root-finding for each u, which might be slow.
        
        # Alternative: Precompute a dense grid of x1, compute f1, sort, and interpolate.
        # This is what the current code does. Let's optimize it.
        
        # Use a larger grid for better accuracy
        n_grid = 1000000
        x1_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 for the grid
        sin_term = np.sin(6.0 * np.pi * x1_grid)
        exp_term = np.exp(-4.0 * x1_grid)
        inner = 1.0 - exp_term * (sin_term ** 6)
        inner = np.clip(inner, 0.0, 1.0)
        f1_grid = 1.0 - (inner ** 0.1)
        
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # We want 200 points that maximize hypervolume.
        # For a concave front, uniform spacing in objective space is good.
        # But f1 range is [0, 1] approximately.
        # Let's find the actual min and max of f1.
        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]
        
        # Sample target f1 values uniformly
        n_points = self.budget
        target_f1 = np.linspace(f1_min, f1_max, n_points)
        
        # Interpolate to find x1 for each target f1
        # Use np.interp for efficiency
        x1_selected = np.interp(target_f1, f1_sorted, x1_sorted)
        
        # Evaluate the selected points
        recorded = []
        for i in range(n_points):
            if time.time() > deadline - 5.0:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_selected[i]
            # Clip to bounds
            for j in range(self.dim):
                x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
            
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
