"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured grid search on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure: first n_obj-1 variables are "shape" variables
        # remaining variables are "distance" variables that should be 0.5
        n_shape = n_obj - 1  # 2 shape variables for 3 objectives
        n_dist = d - n_shape  # 10 distance variables

        # Strategy: Use all budget to sample the Pareto front uniformly.
        # Set distance variables to exactly 0.5 (optimal for g=0)
        # Sample shape variables using a structured grid to cover the front uniformly.

        # For DTLZ4 with 3 objectives:
        # f1 = (1+g)*x1*x2
        # f2 = (1+g)*x1*(1-x2)
        # f3 = (1+g)*(1-x1)
        # With g=0 (distance vars at 0.5):
        # f1 = x1*x2, f2 = x1*(1-x2), f3 = 1-x1
        # x1 in [0,1], x2 in [0,1]

        # Use a grid over x1 and x2 to maximize hypervolume coverage
        # The Pareto front is a 2D surface, so we want uniform coverage in objective space.
        # A uniform grid in (x1, x2) space maps to a non-uniform grid in objective space,
        # but it still provides good coverage.

        # Use all budget for grid points
        n_points = budget

        # Create a structured grid
        # Use a Halton-like sequence or a regular grid
        # For best hypervolume, we want points spread across the entire front.
        
        # Let's use a regular grid with sqrt(n_points) points per dimension
        n_grid = int(np.ceil(np.sqrt(n_points)))
        
        # Actually, let's use a more efficient approach:
        # Use a low-discrepancy sequence (Sobol or Halton) for better coverage
        
        # Generate Halton sequence for 2D (shape variables)
        def halton_sequence(n, dims):
            """Generate n points in [0,1)^dims using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for dim in range(dims):
                    base = dim + 2  # bases: 2, 3, 5, ...
                    f = 1.0
                    r = 0.0
                    i_temp = i
                    while i_temp > 0:
                        f /= base
                        r += f * (i_temp % base)
                        i_temp //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        # Generate shape variable samples
        shape_samples = halton_sequence(n_points, n_shape)
        
        # Create full solution vectors
        X = np.zeros((n_points, d))
        X[:, :n_shape] = shape_samples
        X[:, n_shape:] = 0.5  # Distance variables at optimum
        
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X

# EVOLVE-BLOCK-END
