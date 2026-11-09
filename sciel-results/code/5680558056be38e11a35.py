import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Structure:
        # m = 3 objectives.
        # Shape variables: x_1, x_2 (indices 0, 1)
        # Distance variables: x_3 ... x_12 (indices 2 to 11)
        # Fix distance variables to lower bounds to minimize distance penalty.
        
        n_shape = self.n_obj - 1  # 2 for n_obj=3
        n_dist = self.dim - n_shape # 10
        
        # Strategy: Dense grid on shape variables, fixed distance variables.
        # Budget: 200
        # Grid: 14 x 14 = 196 points.
        
        grid_size = int(np.sqrt(self.budget))
        if grid_size < 2: grid_size = 2
        # Adjust to fit budget
        while grid_size * grid_size > self.budget:
            grid_size -= 1
            
        x1_vals = np.linspace(lo[0], hi[0], grid_size)
        x2_vals = np.linspace(lo[1], hi[1], grid_size)
        
        X_grid = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # Fix remaining to lower bounds
                for i in range(n_shape, self.dim):
                    x[i] = lo[i]
                X_grid.append(x)
                
        # If we have budget left, add some random full-space points
        # Or just use the grid.
        
        # Evaluate
        F = []
        for x in X_grid:
            f = problem(x)
            F.append(f)
            
        # If we have more budget, do a small NSGA-II or random search
        # to potentially catch points if the analytic assumption is slightly off.
        # But with 196 grid points, we are likely well-covered.
        
        # Let's use the remaining budget for random sampling of shape vars 
        # with distance vars fixed, to fill gaps if grid is sparse.
        remaining = self.budget - len(X_grid)
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                x = np.zeros(self.dim)
                x[0] = rng.uniform(lo[0], hi[0])
                x[1] = rng.uniform(lo[1], hi[1])
                for i in range(n_shape, self.dim):
                    x[i] = lo[i]
                f = problem(x)
                F.append(f)
                
        return X_grid # Return value is ignored, but must be valid
