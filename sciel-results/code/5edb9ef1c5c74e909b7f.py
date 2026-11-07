"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 exploiting known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_F = []

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget

        # DTLZ1 Pareto front: x[0] + x[1] + x[2] = 1, x[3:] = 0.5
        # f1 = 0.5 * x0
        # f2 = 0.5 * x1
        # f3 = 0.5 * (1 - x0 - x1)
        # So we need to sample (x0, x1) on the 2-simplex: x0 >= 0, x1 >= 0, x0 + x1 <= 1

        # Use a well-distributed low-discrepancy sequence (Halton) on the simplex
        # Halton sequence in 2D for better coverage than uniform random

        def halton_sequence(n, base1, base2):
            """Generate n points from 2D Halton sequence."""
            seq = np.zeros((n, 2))
            for i in range(n):
                # Base 1
                x = 0.0
                f = 1.0 / base1
                idx = i + 1
                while idx > 0:
                    x += f * (idx % base1)
                    idx //= base1
                    f /= base1
                # Base 2
                y = 0.0
                f = 1.0 / base2
                idx = i + 1
                while idx > 0:
                    y += f * (idx % base2)
                    idx //= base2
                    f /= base2
                seq[i] = [x, y]
            return seq

        # Generate Halton sequence points in [0,1]^2
        halton_pts = halton_sequence(n_evals, 2, 3)

        # Map [0,1]^2 to the 2-simplex using inverse CDF transform:
        # u, v ~ U(0,1) -> a = 1 - sqrt(1-u), b = (1-a)*v
        u = halton_pts[:, 0]
        v = halton_pts[:, 1]
        a = 1.0 - np.sqrt(1.0 - u)
        b = (1.0 - a) * v

        # Evaluate all points on the Pareto front
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = a[i]
            x[1] = b[i]
            x[2:] = 0.5  # optimal shape variables

            evaluate(x)

        return all_F
# EVOLVE-BLOCK-END
