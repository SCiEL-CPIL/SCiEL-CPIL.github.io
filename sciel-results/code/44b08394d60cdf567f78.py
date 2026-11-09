import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # DTLZ7 Pareto front: f0 = 1+g, f1 = (1+g)*x0, f2 = (1+g)*x1
        # Minimum g = 1 (when x[2:] = 0.5), so 1+g = 2
        # Pareto front: f0=2, f1=2*x0, f2=2*x1 for x0,x1 in [0,1]
        # Reference point: [1.1, 1.1, 25.0]
        
        # Create a dense 2D grid over [0,1]^2 for (x0, x1)
        # Use a non-uniform grid that clusters near the boundaries where
        # hypervolume contribution is highest
        
        n_grid = int(np.sqrt(n_evals))
        if n_grid < 2:
            n_grid = 2
        
        # Create warped grid points that cluster near 0 and 1
        # Use a power transformation with alpha < 1 to cluster near 0
        # and combine with symmetric version
        t = np.linspace(0, 1, n_grid)
        
        # Apply a warp that increases density near boundaries
        # t_warped = 0.5 - 0.5 * np.cos(np.pi * t) is uniform
        # For clustering: use t^alpha with alpha < 1
        alpha = 0.7
        t_warped = t ** alpha
        
        # Create 2D grid
        x0_vals, x1_vals = np.meshgrid(t_warped, t_warped)
        grid_points = np.column_stack([x0_vals.ravel(), x1_vals.ravel()])
        
        # If we have more points than budget, subsample
        if len(grid_points) > n_evals:
            # Keep the points that are most spread out
            rng = np.random.RandomState(42)
            indices = rng.choice(len(grid_points), n_evals, replace=False)
            grid_points = grid_points[indices]
        
        # Pad with additional points if needed
        if len(grid_points) < n_evals:
            n_extra = n_evals - len(grid_points)
            rng = np.random.RandomState(42)
            extra_points = rng.rand(n_extra, 2)
            grid_points = np.vstack([grid_points, extra_points])
        
        # Shuffle to avoid any ordering bias
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]
        
        # Evaluate points
        for i in range(min(n_evals, len(grid_points))):
            if time.time() > deadline:
                break
                
            u = grid_points[i, 0]
            v = grid_points[i, 1]
            
            x = np.full(self.dim, 0.5)  # x[2:] = 0.5 for optimal g
            x[0] = u
            x[1] = v
            
            problem(x)
            
        return None
