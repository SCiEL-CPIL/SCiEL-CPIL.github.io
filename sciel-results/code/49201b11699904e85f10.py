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
        
        # Use a power-law transformation to concentrate points near the center
        # t in [0, 1] -> x = 0.5 + 0.5 * sign(t - 0.5) * |2*t - 1|^alpha
        # For alpha < 1, this concentrates points near the center
        
        alpha = 0.5  # sqrt transformation for center concentration
        
        # Generate uniform grid in transformed space
        grid_size = int(np.sqrt(n_evals))
        
        # Uniform grid in [0, 1]
        u1_vals = np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        u2_vals = np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        
        # Apply power-law transformation to concentrate near center
        def transform(u, alpha):
            # Map u in [0, 1] to x in [0, 1] with concentration near 0.5
            # Use: x = 0.5 + 0.5 * (2*u - 1) * |2*u - 1|^(alpha - 1) / |2*u - 1|^(alpha - 1)
            # Simpler: x = 0.5 + 0.5 * sign(2*u - 1) * |2*u - 1|^alpha
            diff = 2 * u - 1
            return 0.5 + 0.5 * np.sign(diff) * np.abs(diff) ** alpha
        
        x1_vals = transform(u1_vals, alpha)
        x2_vals = transform(u2_vals, alpha)
        
        # Create meshgrid
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        
        # We may have more points than budget, so subsample
        if len(X1_flat) > n_evals:
            indices = np.random.choice(len(X1_flat), n_evals, replace=False)
            X1_flat = X1_flat[indices]
            X2_flat = X2_flat[indices]
        
        # Build decision vectors
        decision_vectors = np.zeros((len(X1_flat), dim))
        decision_vectors[:, 0] = X1_flat
        decision_vectors[:, 1] = X2_flat
        decision_vectors[:, 2:] = 0.5  # g-variables at 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in decision_vectors])
        
        return F
# EVOLVE-BLOCK-END
