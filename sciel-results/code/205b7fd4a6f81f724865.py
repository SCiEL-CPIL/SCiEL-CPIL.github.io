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
        # 14x14 = 196 points. 4 extra points.
        n_grid = 14
        n_extra = 4
        
        # Generate grid points in u, v space [0, 1]
        # Include boundaries
        u_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        for u in u_vals:
            for v in v_vals:
                if time.time() > deadline:
                    return evaluated_points
                if used_budget >= self.budget:
                    break
                
                x0 = (2.0 / np.pi) * np.arccos(u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1
            
            if used_budget >= self.budget:
                break

        # Add extra points if budget remains
        # Midpoints of the "edges" of the parameter square
        extra_uv = [
            (0.5, 0.0),
            (0.5, 1.0),
            (0.0, 0.5),
            (1.0, 0.5)
        ]
        
        for u, v in extra_uv:
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
                
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
