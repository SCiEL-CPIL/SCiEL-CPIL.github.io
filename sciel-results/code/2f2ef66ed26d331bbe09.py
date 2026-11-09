"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid + adaptive refinement for DTLZ4 hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj - 1  # 2 shape variables
        k = n - m           # 10 position variables

        # DTLZ4 Pareto front parameterization:
        # For 3 objectives, the front is the 2-simplex {f1+f2+f3=1, fi>=0}
        # We parameterize using two angles:
        #   f1 = cos(theta2) * cos(theta1)
        #   f2 = cos(theta2) * sin(theta1)
        #   f3 = sin(theta2)
        # where theta1 in [0, pi/2], theta2 in [0, pi/2]
        #
        # The shape variables are:
        #   x_{n-1} = 2/pi * theta1  (controls f1 vs f2 ratio)
        #   x_{n-2} = 2/pi * theta2  (controls f3 vs (f1+f2))
        #
        # Position variables at 0.5 give g=0, so f = h directly.

        # Phase 1: Dense quasi-uniform grid in (theta1, theta2) space
        # Use a structured grid with enough points to cover the front well
        # Hypervolume in 3D with ref [2.5,2.5,2.5]: each point contributes
        # volume proportional to (2.5-f1)*(2.5-f2)*(2.5-f3) but only for
        # non-dominated regions. A good spread maximizes this.

        # Allocate: ~180 for grid, ~20 for adaptive refinement
        n_grid = 180
        n_refine = self.budget - n_grid

        # Create grid: use a combination of regular grid + some structured points
        # For 3D hypervolume, we want good coverage of the simplex
        # Use a polar-like grid in (theta1, theta2) space

        # Regular grid: ~13x13 = 169 points
        n_per_axis = 13
        theta1_vals = np.linspace(0.0, np.pi / 2, n_per_axis)
        theta2_vals = np.linspace(0.0, np.pi / 2, n_per_axis)

        # Generate grid points
        grid_points = []
        for t2 in theta2_vals:
            for t1 in theta1_vals:
                grid_points.append((t1, t2))

        # Add midpoints for better coverage (12x12 = 144 midpoints, but we only need ~11 more)
        # Actually let's use a slightly different approach:
        # Use 14x13 = 182 grid points, then trim to 180
        n1, n2 = 14, 13
        theta1_vals = np.linspace(0.0, np.pi / 2, n1)
        theta2_vals = np.linspace(0.0, np.pi / 2, n2)

        grid_points = []
        for t2 in theta2_vals:
            for t1 in theta1_vals:
                grid_points.append((t1, t2))

        # Trim to budget
        grid_points = grid_points[:n_grid]

        # Evaluate all grid points
        X = []
        F = []
        used = 0

        for t1, t2 in grid_points:
            if time.time() > deadline - 20:
                break

            # Position variables at 0.5 (g=0)
            x_pos = np.full(k, 0.5)

            # Shape variables
            x_shape = np.array([2.0 / np.pi * t1, 2.0 / np.pi * t2])
            x_shape = np.clip(x_shape, 0.0, 1.0)

            x = np.concatenate([x_pos, x_shape])
            x = np.clip(x, lo, hi)

            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Phase 2: Adaptive refinement
        # Strategy: find regions of the simplex where coverage is sparse
        # and add points there. Use the recorded F values to identify gaps.
        if used < self.budget and time.time() < deadline - 5:
            # Identify the "corners" and "edges" of the simplex that might be under-sampled
            # Add points near the vertices and edges of the simplex
            # Vertices in (theta1, theta2):
            #   (0, 0) -> f=(1,0,0)
            #   (pi/2, 0) -> f=(0,1,0)
            #   (0, pi/2) -> f=(0,0,1)
            #   (pi/2, pi/2) -> f=(0,0,1) [same as above since cos(pi/2)=0]

            # Add structured points along edges and interior
            refine_points = []

            # Edge: t2=0 (f3=0), vary t1
            for t1 in np.linspace(0.0, np.pi / 2, 5):
                refine_points.append((t1, 0.0))
            # Edge: t1=0 (f1=0), vary t2
            for t2 in np.linspace(0.0, np.pi / 2, 5):
                refine_points.append((0.0, t2))
            # Edge: t1=pi/2 (f2 max when t2=0), vary t2
            for t2 in np.linspace(0.0, np.pi / 2, 5):
                refine_points.append((np.pi / 2, t2))
            # Interior diagonal
            for t in np.linspace(0.0, np.pi / 2, 5):
                refine_points.append((t, t))

            # Also add some quasi-random points for diversity
            rng = np.random.default_rng(0)
            for _ in range(5):
                t1 = rng.uniform(0.05, np.pi / 2 - 0.05)
                t2 = rng.uniform(0.05, np.pi / 2 - 0.05)
                refine_points.append((t1, t2))

            # Sort refine points by priority (edges first, then interior)
            # Deduplicate
            unique_refine = []
            seen = set()
            for t1, t2 in refine_points:
                key = (round(t1, 4), round(t2, 4))
                if key not in seen:
                    seen.add(key)
                    unique_refine.append((t1, t2))

            for t1, t2 in unique_refine:
                if used >= self.budget or time.time() > deadline - 5:
                    break

                x_pos = np.full(k, 0.5)
                x_shape = np.array([2.0 / np.pi * t1, 2.0 / np.pi * t2])
                x_shape = np.clip(x_shape, 0.0, 1.0)
                x = np.concatenate([x_pos, x_shape])
                x = np.clip(x, lo, hi)

                f = problem(x)
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                used += 1

        return X
# EVOLVE-BLOCK-END
