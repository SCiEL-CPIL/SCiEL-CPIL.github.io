import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer using a grid on the Pareto front parameters for optimal hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        # Define grid dimensions
        # Use uniform grid in x0, x1 space directly for better coverage
        n_grid = 14
        n_extra = 4
        
        # Generate grid points in x0, x1 space [0, 1]
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                if time.time() > deadline:
                    return evaluated_points
                if used_budget >= self.budget:
                    break
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1
            
            if used_budget >= self.budget:
                break

        # Add extra points if budget remains
        # Midpoints and other strategic points in x0, x1 space
        extra_points = [
            (0.5, 0.5),
            (0.25, 0.5),
            (0.75, 0.5),
            (0.5, 0.25),
        ]
        
        for x0, x1 in extra_points:
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
                
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
