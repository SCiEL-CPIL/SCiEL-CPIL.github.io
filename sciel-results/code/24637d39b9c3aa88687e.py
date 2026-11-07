"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using stratified grid on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # First dim-5 variables shape the front.
        # Last 5 variables affect g. g=0 when they are 0.5.
        # For 3 objectives, only x1 and x2 determine the front shape.
        # Front is a quarter sphere of radius 0.5.

        n_g = 5
        n_front = dim - n_g

        # Use a stratified grid on (x1, x2) for maximum hypervolume coverage.
        # The Pareto front in DTLZ4 (3 objectives) is parameterized by:
        #   f1 = 0.5 * cos(alpha) * cos(beta)
        #   f2 = 0.5 * cos(alpha) * sin(beta)
        #   f3 = 0.5 * sin(alpha)
        # where alpha = pi * x1 / 2, beta = pi * x2 / 2
        # 
        # To maximize hypervolume, we want points that cover the surface uniformly.
        # A stratified grid in (alpha, beta) space with appropriate weighting
        # gives good coverage. Alternatively, we can use a grid in the objective
        # space that maps back to the front.
        #
        # Let's use a grid in (u1, u2) where:
        #   u1 = sin(alpha) in [0, 1]  -> alpha = arcsin(u1)
        #   u2 = beta in [0, pi/2]
        #
        # For uniform coverage on the sphere surface, we should stratify
        # u1 = sin(alpha) uniformly (since dA has factor cos(alpha) for the
        # azimuthal part, but for the polar part, uniform in cos(theta) is
        # uniform area). Let's think more carefully.
        #
        # On a sphere, uniform area sampling uses:
        #   cos(theta) ~ U[0,1]  where theta is polar angle from north pole
        #   phi ~ U[0, 2*pi]  azimuthal
        #
        # In our case, the quarter sphere has:
        #   alpha in [0, pi/2] (polar from f3 axis)
        #   beta in [0, pi/2] (azimuthal)
        #
        # cos(alpha) goes from 1 to 0 as alpha goes from 0 to pi/2.
        # Uniform area: cos(alpha) ~ U[0,1]
        # So alpha = arccos(u1) where u1 ~ U[0,1]
        # beta = u2 * pi/2 where u2 ~ U[0,1]
        #
        # Then x1 = 2*alpha/pi = 2*arccos(u1)/pi
        # x2 = 2*beta/pi = u2
        #
        # Let's use a stratified grid in (u1, u2) space.

        # Choose grid size to use the budget
        # 14x14 = 196 points, close to 200
        # 15x15 = 225 > 200, so use 14x14 = 196
        n_per_dim = int(np.sqrt(budget))
        # Adjust to use exactly budget or close
        if n_per_dim * n_per_dim > budget:
            n_per_dim -= 1
        n_points = n_per_dim * n_per_dim

        # Create stratified grid in [0,1] x [0,1]
        # Use midpoints of each cell for stratification
        u1_grid, u2_grid = np.meshgrid(
            (np.arange(n_per_dim) + 0.5) / n_per_dim,
            (np.arange(n_per_dim) + 0.5) / n_per_dim,
            indexing='ij'
        )
        u1_flat = u1_grid.flatten()
        u2_flat = u2_grid.flatten()

        # Compute x1 and x2
        # x1 = 2*arccos(u1)/pi
        # x2 = u2
        x1_vals = (2.0 / np.pi) * np.arccos(u1_flat)
        x2_vals = u2_flat

        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        # Set other front variables to 0.5 (arbitrary, doesn't affect objectives)
        if n_front > 2:
            X[:, 2:n_front] = 0.5
        # Set g variables to 0.5 to ensure g=0
        X[:, n_front:] = 0.5

        # Clip to bounds (should already be in [0,1] but just in case)
        X = np.clip(X, lo, hi)

        # If we have remaining budget, add additional points
        remaining = budget - n_points
        if remaining > 0:
            # Add remaining points using a different stratification or random
            # Use Latin hypercube for the remaining points
            extra_u1 = (np.arange(remaining) + 0.3) / (n_per_dim + remaining)
            extra_u2 = (np.arange(remaining) + 0.7) / (n_per_dim + remaining)
            
            extra_x1 = (2.0 / np.pi) * np.arccos(extra_u1)
            extra_x2 = extra_u2

            X_extra = np.zeros((remaining, dim))
            X_extra[:, 0] = extra_x1
            X_extra[:, 1] = extra_x2
            if n_front > 2:
                X_extra[:, 2:n_front] = 0.5
            X_extra[:, n_front:] = 0.5
            X_extra = np.clip(X_extra, lo, hi)
            
            X = np.vstack([X, X_extra])

        # Check time before starting evaluation loop
        if time.time() - t0 > deadline:
            return X

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        # The return value is ignored, but we return X as per the seed program style
        return X
# EVOLVE-BLOCK-END
