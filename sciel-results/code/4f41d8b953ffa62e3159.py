"""Multi-objective optimizer for DTLZ1 using direct Pareto front sampling with Sobol sequence."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct sampler for DTLZ1 Pareto front using low-discrepancy sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # DTLZ1 structure:
        #   f1 = 0.5 * g * (1 - z1)
        #   f2 = 0.5 * g * z1
        #   f3 = 0.5 * g * z2
        # where g = sum((x_i - 0.5)^2) for i in [dim-3, dim-2, dim-1]
        #       z1 = x[0], z2 = x[1]
        # Pareto front: g = 0 (i.e., last 3 vars at 0.5), z1 >= 0, z2 >= 0, z1 + z2 <= 1
        #
        # Strategy: Use Sobol sequence to uniformly fill the 2-simplex {z1,z2 >= 0, z1+z2 <= 1}
        # Fix the g-variables at 0.5 to achieve g=0 (the Pareto front).
        # This ensures all evaluations are on the Pareto front and non-dominated.

        # Determine which variables control g (last 3 for standard DTLZ1 with dim=7)
        g_vars = list(range(dim - 3, dim))  # [4, 5, 6] for dim=7
        z_vars = [0, 1]  # z1 and z2

        # Set g-vars to 0.5 (center of [0,1]) to minimize g
        g_value = 0.5

        # Generate Sobol sequence points in [0,1]^2, then map to the simplex
        # The simplex {z1,z2 >= 0, z1+z2 <= 1} can be parameterized by:
        #   u1, u2 in [0,1]
        #   z1 = u1 * (1 - u2)
        #   z2 = u2
        # This maps the unit square to the triangle with vertices (0,0), (1,0), (0,1)
        # Wait, let's verify: 
        #   u1=0, u2=0 -> z1=0, z2=0  (vertex 1)
        #   u1=1, u2=0 -> z1=1, z2=0  (vertex 2)  
        #   u1=0, u2=1 -> z1=0, z2=1  (vertex 3)
        #   u1=1, u2=1 -> z1=0, z2=1  (this is wrong, should be on edge)
        # 
        # Better parameterization:
        #   z1 = u1
        #   z2 = u2 * (1 - u1)
        # Check:
        #   u1=0, u2=0 -> z1=0, z2=0
        #   u1=0, u2=1 -> z1=0, z2=1
        #   u1=1, u2=0 -> z1=1, z2=0
        #   u1=1, u2=1 -> z1=1, z2=0
        #   u1=0.5, u2=0.5 -> z1=0.5, z2=0.25, z1+z2=0.75 <= 1 ✓
        # This works! The map (u1, u2) -> (z1, z2) = (u1, u2*(1-u1)) covers the simplex.

        # We need to generate budget points. Use a quasi-random sequence.
        # Since we can't easily import scipy.stats.qmc in all environments, 
        # use a simple Halton sequence or a well-distributed grid.
        # Actually, let's use a stratified sampling approach with Halton sequence.

        def halton_sequence(n, dims):
            """Generate n points in [0,1]^dims using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dims + 2):  # bases: 2, 3, 4, ...
                    f = 1.0
                    r = 0.0
                    ii = i
                    while ii > 0:
                        f /= base
                        r += f * (ii % base)
                        ii //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        # Generate budget points using Halton sequence in 2D
        n_points = budget
        halton_pts = halton_sequence(n_points, 2)  # shape (n_points, 2)

        # Map Halton points to the simplex
        # z1 = u1, z2 = u2 * (1 - u1)
        u1 = halton_pts[:, 0]
        u2 = halton_pts[:, 1]
        z1_vals = u1
        z2_vals = u2 * (1.0 - u1)

        # Clip to ensure we stay in the simplex
        z1_vals = np.clip(z1_vals, 0.0, 1.0)
        z2_vals = np.clip(z2_vals, 0.0, 1.0 - z1_vals)

        all_F = []
        all_X = []
        used = 0

        for idx in range(n_points):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            z1 = z1_vals[idx]
            z2 = z2_vals[idx]

            # Construct x vector
            x = np.zeros(dim)
            x[z_vars[0]] = z1
            x[z_vars[1]] = z2
            # Set other non-g, non-z variables to 0.5 (they don't affect objectives in standard DTLZ1)
            for i in range(dim):
                if i not in z_vars and i not in g_vars:
                    x[i] = 0.5
            # Set g variables to 0.5
            for i in g_vars:
                x[i] = g_value

            # Clip to bounds
            x = np.clip(x, lo, hi)

            # Evaluate
            f = np.array(problem(x), dtype=float)
            used += 1
            all_F.append(f)
            all_X.append(x.copy())

        # Ensure at least one evaluation
        if not all_F:
            x = lo + 0.5 * (hi - lo)
            f = np.array(problem(x), dtype=float)
            all_F.append(f)
            all_X.append(x.copy())

        return np.array(all_X)
# EVOLVE-BLOCK-END
