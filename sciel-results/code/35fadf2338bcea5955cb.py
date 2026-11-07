import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        
        # Use a grid to ensure corner coverage
        # 14x14 = 196 points, close to 200
        n_grid = int(np.sqrt(n_evals))
        if n_grid * n_grid > n_evals:
            n_grid -= 1
            
        # Generate grid points in [0, 1]
        # np.linspace includes endpoints by default
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        
        # Flatten
        X0_flat = X0.flatten()
        X1_flat = X1.flatten()
        
        n_points = len(X0_flat)
        
        # If we have room, add Sobol points for the remaining budget
        X = np.zeros((n_points, self.dim))
        X[:, 0] = X0_flat
        X[:, 1] = X1_flat
        
        if self.dim > 2:
            mid_points = (lo[2:] + hi[2:]) / 2.0
            X[:, 2:] = mid_points
            
        # Evaluate
        eval_count = 0
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break
                
        return X
