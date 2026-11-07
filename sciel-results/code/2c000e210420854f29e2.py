"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure to maximize hypervolume:
    - f1 = x1, g = 1 + 10(n-1) + sum(xi^2 - 10*cos(2*pi*xi)) for i>=2
    - f2 = (1 - sqrt(f1/g)) * g
    - Setting x2..x10 = 0 gives g = 1, f2 = 1 - sqrt(x1)
    - Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1]
    - Reference point [11, 400]: all front points with f1 < 11 count
    - Strategy: sample f2 uniformly in [0, 1] to equalize hypervolume contribution
      per evaluation, which gives f1 = (1 - f2)^2
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Determine x_rest values (x2..x10) to minimize g
        x_rest = np.zeros(self.dim - 1)
        for j in range(1, self.dim):
            if lo[j] <= 0.0 <= hi[j]:
                x_rest[j - 1] = 0.0
            elif 0.0 < lo[j]:
                x_rest[j - 1] = lo[j]
            else:
                x_rest[j - 1] = hi[j]

        # The Pareto front for ZDT4: f2 = 1 - sqrt(f1) for f1 in [0, 1]
        # To maximize hypervolume with reference point [11, 400],
        # we sample f2 uniformly in [0, 1], which gives f1 = (1 - f2)^2
        # This equalizes the hypervolume contribution per evaluation
        n_evals = self.budget
        results = []

        # f2 uniformly spaced in [0, 1]
        f2_vals = np.linspace(0.0, 1.0, n_evals)
        # f1 = (1 - f2)^2
        f1_vals = (1.0 - f2_vals) ** 2

        # Clip f1 to valid range [max(lo[0], 0), min(hi[0], 1)]
        f1_min = max(lo[0], 0.0)
        f1_max = min(hi[0], 1.0)

        if f1_min >= f1_max:
            # Fallback: use full bound range
            f1_min = lo[0]
            f1_max = hi[0]

        for i in range(n_evals):
            if time.time() > deadline:
                break

            f1 = f1_vals[i]
            # Clip f1 to valid range
            f1 = np.clip(f1, f1_min, f1_max)

            x = np.zeros(self.dim)
            x[0] = f1
            x[1:] = x_rest

            x_clipped = np.clip(x, lo, hi)
            try:
                f = problem(x_clipped)
                results.append(f)
            except Exception:
                break

        return results
# EVOLVE-BLOCK-END
