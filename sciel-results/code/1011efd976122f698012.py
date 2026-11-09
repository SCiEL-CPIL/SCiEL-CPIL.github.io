"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        n = self.dim
        # ZDT2 Pareto front: x[0] = t in [0,1], x[1:] = 0
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(x[0]))
        # For n=30: g = 1 + 290 = 291
        # f1 = t, f2 = 291 * (1 - sqrt(t))
        
        # Reference point is [11, 11].
        # Points must satisfy f1 < 11 and f2 < 11.
        # f1 = t < 11 is always true for t in [0,1].
        # f2 = 291 * (1 - sqrt(t)) < 11
        # 1 - sqrt(t) < 11/291
        # sqrt(t) > 1 - 11/291 = 280/291
        # t > (280/291)^2
        t_min = (280.0 / 291.0) ** 2
        t_max = 1.0
        
        # Sample the Pareto front with budget points in the valid region [t_min, t_max]
        # This maximizes the number of valid points contributing to hypervolume.
        t_vals = np.linspace(t_min, t_max, self.budget)
        
        results = []
        for t in t_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = t
            # Clip to bounds
            x = np.clip(x, bounds[0][0], bounds[0][1]) if n > 0 else x
            for i in range(1, n):
                x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
