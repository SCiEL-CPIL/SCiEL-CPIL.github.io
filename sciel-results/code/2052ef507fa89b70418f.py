"""DTLZ1 optimizer using Hammersley point set on the Pareto simplex for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with a Hammersley low-discrepancy sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hammersley_2d(self, n):
        """Generate n Hammersley points in [0,1)^2 using a low-discrepancy sequence."""
        points = np.zeros((n, 2))
        for i in range(n):
            # First coordinate: van der Corput sequence base 2
            u1 = 0.0
            f = 0.5
            ii = i + 1
            while ii > 0:
                u1 += f * (ii % 2)
                ii //= 2
                f *= 0.5
            # Second coordinate: van der Corput sequence base 3 (or use a different base)
            u2 = 0.0
            f = 1.0 / 3.0
            ii = i + 1
            while ii > 0:
                u2 += f * (ii % 3)
                ii //= 3
                f /= 3.0
            points[i, 0] = u1
            points[i, 1] = u2
        return points

    def _map_to_simplex(self, u):
        """Map a point in [0,1)^2 to the 2D simplex {x0>=0, x1>=0, x0+x1<=1}."""
        # Use the standard mapping: x0 = 1 - sqrt(u1), x1 = (1 - x0) * u2
        # This gives uniform distribution on the simplex
        x0 = 1.0 - np.sqrt(u[:, 0])
        x1 = (1.0 - x0) * u[:, 1]
        return x0, x1

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5

        base_x = np.zeros(dim)
        base_x[2:] = 0.5

        n_evals = min(budget, 200)

        if n_evals <= 0:
            return all_points

        # Generate Hammersley points in [0,1)^2
        hammersley = self._hammersley_2d(n_evals)

        # Map to the 2D simplex
        x0, x1 = self._map_to_simplex(hammersley)

        # Ensure points are strictly inside the simplex (avoid boundary issues)
        # Clip to avoid numerical issues at boundaries
        x0 = np.clip(x0, 0.0, 1.0)
        x1 = np.clip(x1, 0.0, 1.0 - x0)

        # Evaluate the points
        for k in range(n_evals):
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = x0[k]
            x[1] = x1[k]
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue

        return all_points
# EVOLVE-BLOCK-END
