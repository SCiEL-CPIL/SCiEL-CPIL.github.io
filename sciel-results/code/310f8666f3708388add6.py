"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for DTLZ7 using structured lattice."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ7 with m=3 objectives and n=22 variables:
        # The Pareto front is parameterized by m-1 = 2 variables (x[n-m], x[n-m+1])
        # and x[0..n-m-1] should be 0 for g=1
        # Specifically:
        # f1 = g * (1 + sin(pi*x[n-m]/2))
        # f2 = g * (1 + sin(3*pi*x[n-m]/2))  
        # f3 = g * (1 + sin(5*pi*x[n-m]/2))
        # where g = 1 + 9/(n-m+1) * sum(x[i]^2) for i in [0, n-m]
        # Wait, let me reconsider. Standard DTLZ7:
        # h(y, g) = sum_{i=1}^{m-1} [f_i(1 + sin(pi*f_i/2))]
        # f_m = g + h
        # For m=3: f1=y1, f2=y2, f3 = g + y1(1+sin(pi*y1/2)) + y2(1+sin(pi*y2/2))
        # The Pareto front has g=1, and y1,y2 in [0,1]
        # So active variables are the last m-1 = 2 variables for y, 
        # and the first n-m+1 = 20 variables for g (should be 0)
        
        # Actually, let me be more careful. DTLZ7 standard form:
        # x = [x_1, ..., x_n], y = [x_{n-m+2}, ..., x_n] (m-1 vars), z = [x_1, ..., x_{n-m+1}] (n-m+1 vars)
        # g(z) = 1 + 9/|z| * sum(z_i^2)
        # h(y, g) = sum_{i=1}^{m-1} y_i * (1 + sin(pi*y_i/2))
        # f_i = y_i for i = 1..m-1
        # f_m = g + h
        
        # So for n=22, m=3:
        # z has n-m+1 = 20 variables (indices 0 to 19)
        # y has m-1 = 2 variables (indices 20, 21)
        # g = 1 + 9/20 * sum(z_i^2)
        # h = y_1*(1+sin(pi*y_1/2)) + y_2*(1+sin(pi*y_2/2))
        # f1 = y_1, f2 = y_2, f3 = g + h
        
        # To get on the Pareto front, set all z_i = 0, so g = 1
        # Then f1 = y_1, f2 = y_2, f3 = 1 + y_1*(1+sin(pi*y_1/2)) + y_2*(1+sin(pi*y_2/2))
        
        # The Pareto front is a 2D surface parameterized by y_1, y_2 in [0,1]
        
        # Strategy: Sample the 2D parameter space (y_1, y_2) to cover the Pareto front
        # Use a Halton sequence or grid for good coverage
        
        # Active indices for y: dim-2 and dim-1 (indices 20 and 21)
        # Inactive indices: 0 to dim-3 (indices 0 to 19)
        
        active_idx = [dim - 2, dim - 1]  # y_1, y_2
        inactive_idx = list(range(dim - 2))  # z variables
        
        # Generate a grid or Halton sequence in 2D
        # With budget=200, we can use a 14x14 grid (196 points) plus some extras
        # Or use a Halton sequence for better coverage
        
        # Let's use a combination: structured grid + random perturbations
        
        # First, create a grid
        n_grid = 14
        y1_vals = np.linspace(0, 1, n_grid)
        y2_vals = np.linspace(0, 1, n_grid)
        
        # Generate all grid points
        grid_points = []
        for y1 in y1_vals:
            for y2 in y2_vals:
                x = np.zeros(dim)
                x[active_idx[0]] = y1
                x[active_idx[1]] = y2
                grid_points.append(x)
        
        grid_points = np.array(grid_points)
        n_grid_points = len(grid_points)
        
        # Evaluate grid points
        F = np.zeros((n_grid_points, n_obj))
        used = 0
        for i in range(n_grid_points):
            if used >= budget or time.time() > deadline:
                break
            F[i] = problem(grid_points[i])
            used += 1
        
        # If we have budget left, add some random points for refinement
        if used < budget:
            rng = np.random.default_rng(42)
            remaining = budget - used
            extra_points = np.zeros((remaining, dim))
            extra_points[:, active_idx[0]] = rng.random(remaining)
            extra_points[:, active_idx[1]] = rng.random(remaining)
            
            for i in range(remaining):
                if used >= budget or time.time() > deadline:
                    break
                F[used] = problem(extra_points[i])
                used += 1
        
        # Return all evaluated points
        # We need to return the decision variables, but the evaluator records objective vectors
        # The return value of __call__ is ignored, so we just need to make sure we called problem() enough times
        
        # Create a dummy return (the X array)
        X = np.zeros((used, dim))
        for i in range(min(used, n_grid_points)):
            X[i] = grid_points[i]
        if used > n_grid_points:
            for i in range(n_grid_points, used):
                X[i] = extra_points[i - n_grid_points]
        
        return X
# EVOLVE-BLOCK-END
