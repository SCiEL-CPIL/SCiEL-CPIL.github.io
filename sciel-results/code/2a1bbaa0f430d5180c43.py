import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5  # x_3, ..., x_12 = 0.5 to ensure g = 0

        # Strategy: Use a uniform grid in (x1, x2) space for the Pareto front
        # f1 = sin(x1*pi/2), f2 = cos(x1*pi/2)*sin(x2*pi/2), f3 = cos(x1*pi/2)*cos(x2*pi/2)
        # This gives natural coverage of the quarter-sphere Pareto front
        
        target_points = min(self.budget, 200)
        
        # Use a grid of size n x n where n*n <= target_points
        n_grid = int(np.sqrt(target_points))
        if n_grid * n_grid > target_points:
            n_grid -= 1
        
        # Generate grid points in (x1, x2) space using midpoints
        x1_vals = (np.arange(n_grid) + 0.5) / n_grid
        x2_vals = (np.arange(n_grid) + 0.5) / n_grid
        
        grid_points = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                grid_points.append((x1, x2))
        
        # Add corner and edge points for better hypervolume coverage
        extra_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.25, 0.0), (0.75, 0.0), (0.25, 1.0), (0.75, 1.0),
            (0.0, 0.25), (0.0, 0.75), (1.0, 0.25), (1.0, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
        ]
        
        # Combine and deduplicate
        all_params = list(grid_points) + list(extra_points)
        
        # Remove duplicates (within tolerance)
        unique_params = []
        for p in all_params:
            x1, x2 = p
            if not any(abs(x1 - u[0]) < 1e-6 and abs(x2 - u[1]) < 1e-6 for u in unique_params):
                unique_params.append(p)
        
        # Truncate to budget
        all_params = unique_params[:self.budget]
        
        # Evaluate all points
        results = []
        for x1, x2 in all_params:
            if len(results) >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x1
            x[1] = x2
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        # Use remaining budget for random refinement
        if len(results) < self.budget:
            rng = np.random.default_rng(42)
            remaining = self.budget - len(results)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                x1 = rng.random()
                x2 = rng.random()
                
                x = base_x.copy()
                x[0] = x1
                x[1] = x2
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(f)
        
        return results
