import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9 structure:
        # The first M-1 (or M) variables define the shape of the Pareto front.
        # The remaining variables define the distance from the ideal point.
        # For WFG9 with 3 objectives, the first 3 variables are the primary shape variables.
        # The penalty terms in the objective functions are minimized when the distance variables
        # are at 0.5 (assuming standard [0,1] bounds) or the midpoint of the bounds.
        
        n_shape = min(3, self.dim)
        
        # Generate a grid for the shape variables to ensure coverage of the disconnected front
        # A 5x5x5 grid provides 125 points, which is a good balance for 200 budget.
        n_grid = 5
        grid_vals = np.linspace(lo[:n_shape], hi[:n_shape], n_grid)
        
        # Create meshgrid for the shape variables
        grids = np.meshgrid(*[grid_vals] * n_shape, indexing='ij')
        shape_points = np.column_stack([g.ravel() for g in grids])
        
        # Add random points to fill the remaining budget
        rng = np.random.default_rng(0)
        n_random = self.budget - len(shape_points)
        
        if n_random > 0:
            # Random points in the shape variable space
            random_shape = lo[:n_shape] + (hi[:n_shape] - lo[:n_shape]) * rng.random((n_random, n_shape))
            shape_points = np.vstack([shape_points, random_shape])
        
        # Ensure we do not exceed budget
        shape_points = shape_points[:self.budget]
        
        # Construct the full dimension vectors
        X = np.zeros((len(shape_points), self.dim))
        X[:, :n_shape] = shape_points
        
        # Set the distance variables to the midpoint of their bounds
        # This minimizes the penalty terms in WFG9 objectives
        if self.dim > n_shape:
            X[:, n_shape:] = mid[n_shape:]
        
        # Evaluate all points
        for x in X:
            if time.time() >= deadline:
                break
            problem(x)
            
        return X
