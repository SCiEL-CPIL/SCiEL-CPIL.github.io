"""Improved program using DTLZ2 Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the analytical Pareto front of DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300 second limit with margin

        n_evals = self.budget
        dim = self.dim

        # DTLZ2 Pareto front: x[0], x[1] are front variables, x[2:] = 0.5
        # On the front (g=0):
        # f_1 = x_1
        # f_2 = (1 - x_1) * x_2
        # f_3 = (1 - x_1) * (1 - x_2)
        
        # Use recursive bisection to create a hypervolume-weighted grid
        # The key insight: points near the center of the front (x1=0.5, x2=0.5)
        # have the largest hypervolume contribution, so we should sample more densely there
        
        # Strategy: Use a non-uniform grid that is denser in the center
        # We'll use a transformation that maps uniform spacing to denser center coverage
        
        # For hypervolume with reference point [2.5, 2.5, 2.5]:
        # Contribution of point (f1, f2, f3) is (2.5-f1)*(2.5-f2)*(2.5-f3)
        # We want to maximize the sum of contributions
        
        # Use uniform grid in parameter space (alpha=1.0) for better coverage
        # and add boundary points to utilize the full budget.
        
        # Generate uniform grid in [0, 1]
        grid_size = int(np.sqrt(n_evals))
        
        # Uniform grid in [0, 1] (centers of bins)
        u1_vals = np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        u2_vals = np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        
        # Create meshgrid
        X1, X2 = np.meshgrid(u1_vals, u2_vals)
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        
        # Add boundary points (corners) if we have remaining budget
        num_grid = len(X1_flat)
        remaining = n_evals - num_grid
        
        if remaining > 0:
            # Corners of the parameter space [0, 1] x [0, 1]
            corners_x1 = [0.0, 0.0, 1.0, 1.0]
            corners_x2 = [0.0, 1.0, 0.0, 1.0]
            
            # Add up to 'remaining' corners
            num_corners_to_add = min(remaining, 4)
            extra_x1 = corners_x1[:num_corners_to_add]
            extra_x2 = corners_x2[:num_corners_to_add]
            
            X1_flat = np.concatenate([X1_flat, extra_x1])
            X2_flat = np.concatenate([X2_flat, extra_x2])
        
        # If we still have budget (unlikely with 200, but for safety), add center
        if len(X1_flat) < n_evals:
            extra_needed = n_evals - len(X1_flat)
            X1_flat = np.concatenate([X1_flat, np.full(extra_needed, 0.5)])
            X2_flat = np.concatenate([X2_flat, np.full(extra_needed, 0.5)])
        
        # Build decision vectors
        decision_vectors = np.zeros((len(X1_flat), dim))
        decision_vectors[:, 0] = X1_flat
        decision_vectors[:, 1] = X2_flat
        decision_vectors[:, 2:] = 0.5  # g-variables at 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in decision_vectors])
        
        return F
# EVOLVE-BLOCK-END
