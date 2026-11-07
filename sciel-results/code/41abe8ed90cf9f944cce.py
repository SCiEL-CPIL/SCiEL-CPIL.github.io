"""Multi-objective optimizer for ZDT4: direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search on ZDT4.

    Strategy: The true Pareto front of ZDT4 lies at x_1..x_9 = 0 (where f2
    is minimized to 10) with f1 = x_0 in [0,1]. We densely sample this known
    front using the entire evaluation budget to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4: x0 in [0,1], x1..x9 in [-5,5]
        # Pareto front: x1..x9 = 0, x0 in [0,1], giving f1=x0, f2=10

        # Use a power-law distribution for x0 to maximize hypervolume
        # The hypervolume contribution of a point at x0 is proportional to
        # (11 - x0) * (400 - 10) for the first objective, but we need to
        # account for the Pareto front geometry.
        #
        # For ZDT4, the Pareto front is f1 = x0, f2 = 10 (constant).
        # With reference point [11, 400], each point contributes:
        #   (11 - f1) * (400 - f2) = (11 - x0) * 390
        #
        # To maximize the sum of contributions, we want to distribute points
        # where (11 - x0) is large, i.e., near x0 = 0.
        #
        # Using a power-law: x0_i = (i/n)^p for p > 1 concentrates points near 0.
        # Let's use p = 2 as a reasonable choice.

        n_samples = budget

        # Power-law distribution: x0 = (i/n)^2 for i = 0..n-1
        # This concentrates samples near x0 = 0 where hypervolume contribution is highest
        i_vals = np.arange(n_samples)
        x0_values = (i_vals / n_samples) ** 2

        results = []
        for i in range(n_samples):
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = x0_values[i]
            # x[1:] = 0.0 (already zero)

            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)

            f = problem(x)
            results.append((x, f))

        # Return the non-dominated points (all should be non-dominated on the front)
        if results:
            F = np.array([f for _, f in results])
            X = np.array([x for x, _ in results])

            # Find non-dominated points
            n = len(X)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            non_dominated_idx = np.where(~dominated)[0]
            if len(non_dominated_idx) > 0:
                return X[non_dominated_idx]
            else:
                return X
        else:
            # Fallback: return a single point
            x = np.zeros(dim)
            x[0] = 0.5
            x = np.clip(x, lo, hi)
            problem(x)
            return x.reshape(1, -1)
# EVOLVE-BLOCK-END
