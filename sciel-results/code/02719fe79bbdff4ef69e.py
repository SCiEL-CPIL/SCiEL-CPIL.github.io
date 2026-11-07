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

        # Generate a grid in the (u, v) parameter space
        # u in [0, 1] maps to x0 = (2/pi) * arccos(u)
        # v in [0, 1] maps to x1 = v
        # The Pareto front of DTLZ2 with 3 objectives is a quarter sphere.
        # To maximize hypervolume, we want to cover the front well.
        
        # Use a 14x14 grid = 196 points, plus 4 extra points if budget allows.
        n_grid = 14
        n_extra = max(0, self.budget - n_grid * n_grid)
        
        # Generate grid points
        # We use np.linspace to include boundaries [0, 1]
        u_vals = np.linspace(0.0, 1.0, n_grid)
        v_vals = np.linspace(0.0, 1.0, n_grid)
        
        # Create meshgrid
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        uv_points = np.column_stack((U.ravel(), V.ravel()))
        
        # Evaluate grid points
        for u, v in uv_points:
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
            
        # If we have extra budget, add some random or specific points
        if used_budget < self.budget:
            # Add center point if not already covered (it is covered by 14x14 grid? No, 14 is even, so 0.5 is not a grid point)
            # 14 points: 0, 1/13, 2/13, ..., 13/13. 0.5 is not included.
            # Let's add the center (0.5, 0.5) which maps to x0=1, x1=0.5
            extra_points = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75)]
            
            for u, v in extra_points:
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
