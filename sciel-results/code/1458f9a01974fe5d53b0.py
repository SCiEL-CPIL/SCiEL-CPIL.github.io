import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Grid Search
        # Use 3 points per dimension to cover the space
        # Points: -5, 0, 5
        grid_vals = np.array([-5.0, 0.0, 5.0])
        
        # Generate all combinations
        # Using np.meshgrid
        axes = [grid_vals] * self.dim
        mesh = np.meshgrid(*axes)
        x_grid = np.vstack([m.ravel() for m in mesh]).T
        
        # Evaluate all grid points
        f_grid = np.array([func(x) for x in x_grid])
        
        # Find the best point
        best_idx = np.argmin(f_grid)
        x = x_grid[best_idx].copy()
        fx = f_grid[best_idx]
        
        # Phase 2: Local Search (1+1)-ES
        remaining_budget = self.budget - len(x_grid)
        
        if remaining_budget > 0:
            rng = np.random.default_rng(0)
            # Start with a smaller sigma as we are likely close to a local minimum
            sigma = 0.5 
            
            for _ in range(remaining_budget):
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                    
        return x, fx
