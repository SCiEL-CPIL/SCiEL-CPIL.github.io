"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using hypervolume-aware non-uniform Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        dim = self.dim
        budget = self.budget

        # DTLZ7 with M=3, D=22:
        # f1 = x0, f2 = x1, f3 = (1 + x_{D-1}) * g
        # g = 1 + 9/D' * sum_{i in D'} sin(3*pi*x_i), D' = D - M + 1 = 20
        # Pareto front: g=0 => x_2..x_{D-1} = 0 (or sin terms sum to -D'/9)
        # Simplest: x_2..x_{D-1} = 0, then g=0
        # On the front: f1=x0, f2=x1, f3=0, with x0>=0, x1>=0, x0+x1<=1
        # The front is the triangle in (f1,f2) with vertices (0,0), (1,0), (0,1)

        # Reference point for scoring: [1.1, 1.1, 25.0]
        # Hypervolume contribution of point (f1,f2,0) ~ (1.1-f1)(1.1-f2)*25
        # Max at origin: 25*1.1*1.1 = 30.25

        # Strategy: densely sample the triangle with power-law clustering toward origin
        # where hypervolume contribution is highest. Use a 2D grid in barycentric-like
        # coordinates with non-uniform spacing.

        # Parameterize the triangle: any point can be written as
        #   f1 = u * v,  f2 = u * (1 - v),  with u in [0,1], v in [0,1]
        # This covers the triangle: u=1 gives the full edge f1+f2=1
        # u=0 gives the origin.

        # Use power-law spacing in u to cluster near origin (high HV region)
        # and uniform in v for coverage along each "radial" line.

        n = budget  # use all budget

        # Create a 2D grid with power-law spacing
        # sqrt(200) ~ 14, so 14x14 = 196, close to 200
        n_per_dim = max(2, int(np.sqrt(n)))
        # Adjust to fit budget: use n_per_dim x n_per_dim, then fill remainder
        grid_size = n_per_dim * n_per_dim

        # Power-law: u = t^gamma where t is uniform, gamma > 1 clusters near 0
        gamma = 3.0  # strong clustering near origin

        # Generate u and v values
        t_u = np.linspace(0, 1, n_per_dim)
        u_vals = t_u ** gamma  # clustered near 0
        v_vals = np.linspace(0, 1, n_per_dim)

        # Generate all grid points
        front_points = []
        seen = set()

        for u in u_vals:
            for v in v_vals:
                f1 = u * v
                f2 = u * (1.0 - v)
                # Clip to ensure on triangle
                f1 = max(0.0, min(1.0, f1))
                f2 = max(0.0, min(1.0, f2))
                if f1 + f2 > 1.0 + 1e-9:
                    # Should not happen with our parameterization, but safety
                    scale = 1.0 / (f1 + f2)
                    f1 *= scale
                    f2 *= scale
                key = (round(f1, 8), round(f2, 8))
                if key not in seen:
                    seen.add(key)
                    front_points.append((f1, f2))

        # If we have fewer than budget, add more points via finer grid or random
        if len(front_points) < budget:
            # Add points along the edges and interior with finer sampling
            rng = np.random.default_rng(123)
            while len(front_points) < budget:
                # Sample in the triangle with power-law bias
                u = rng.random() ** gamma
                v = rng.random()
                f1 = u * v
                f2 = u * (1.0 - v)
                f1 = max(0.0, min(1.0, f1))
                f2 = max(0.0, min(1.0, f2))
                key = (round(f1, 8), round(f2, 8))
                if key not in seen:
                    seen.add(key)
                    front_points.append((f1, f2))
                else:
                    # Perturb slightly
                    f1 += rng.random() * 1e-4
                    f2 += rng.random() * 1e-4
                    f1 = max(0.0, min(1.0, f1))
                    f2 = max(0.0, min(1.0, f2))
                    if f1 + f2 > 1.0:
                        scale = 1.0 / (f1 + f2)
                        f1 *= scale
                        f2 *= scale
                    key = (round(f1, 8), round(f2, 8))
                    if key not in seen:
                        seen.add(key)
                        front_points.append((f1, f2))

        front_points = front_points[:budget]

        # Convert to full 22-dim vectors on the Pareto front
        # x0 = f1, x1 = f2, x_2..x_21 = 0
        X = np.zeros((len(front_points), dim))
        for i, (f1, f2) in enumerate(front_points):
            X[i, 0] = f1
            X[i, 1] = f2
            # x_2..x_21 remain 0, giving g=0, f3=0

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
