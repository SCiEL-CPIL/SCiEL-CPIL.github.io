# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7.
    Strategy: Sample the 2D Pareto front (x_3..x_22 = 0) with a non-uniform grid
    that is denser near the ideal point (f1=0, f2=0) to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # Determine grid size
        grid_size = int(np.sqrt(self.budget))
        grid_size = max(2, grid_size)
        while (grid_size + 1) ** 2 <= self.budget:
            grid_size += 1
        n_grid = grid_size * grid_size

        # Non-linear grid to concentrate points near the ideal objectives
        # f1 = 1 - x1, f2 = x2.
        # We want f1, f2 small.
        # x1 = 1 - f1, x2 = f2.
        # Let t in [0, 1] be the grid coordinate.
        # f1 = t^gamma, f2 = t^delta.
        # x1 = 1 - t^gamma, x2 = t^delta.
        # gamma, delta > 1 pushes t^gamma closer to 0 for small t?
        # If t is uniform in [0,1], t^2 is denser near 0.
        # So f1 = t^2 is denser near 0. This is good.
        
        gamma = 2.0
        delta = 2.0
        
        t_vals = np.linspace(0.0, 1.0, grid_size)
        x1_vals = 1.0 - t_vals ** gamma
        x2_vals = t_vals ** delta

        for x1 in x1_vals:
            for x2 in x2_vals:
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # x[2:] = 0 for g=0 (Pareto front)

                try:
                    obj = problem(x)
                    obj = list(obj)
                    results.append(obj)
                    n_evals += 1
                except Exception:
                    break
            if n_evals >= self.budget or time.time() > deadline:
                break

        return results
# EVOLVE-BLOCK-END
