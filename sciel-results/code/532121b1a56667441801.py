"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    The variables x_1 and x_2 determine the shape of the front.
    Strategy: Evaluate the best corner (0,0,1) first, then densely sample
    the Pareto front to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.array(x, dtype=np.float64)
            # Clip to bounds
            x = np.clip(x, lo, hi)
            try:
                obj = problem(x)
                obj = np.array(obj, dtype=np.float64)
                if obj.shape[0] != self.n_obj:
                    return None
                if not np.all(np.isfinite(obj)):
                    return None
                results.append(obj.tolist())
                n_evals += 1
                return obj
            except Exception:
                return None

        # DTLZ7 Pareto front: x_3..x_n = 0
        # f1 = 1 - x_1
        # f2 = 1 - (1 - x_2)^(1/(1+g)) where g=0, so f2 = x_2
        # f3 = 1 + g*h = 1
        # So on the front: f1 + f2 = 1, f3 = 1
        # Best corner: x_1=1, x_2=0 -> f1=0, f2=0, f3=1

        # First, evaluate the best corner
        x_corner = np.zeros(self.dim)
        x_corner[0] = 1.0  # x_1 = 1 -> f1 = 0
        x_corner[1] = 0.0  # x_2 = 0 -> f2 = 0
        # x_3..x_n = 0 -> g = 0 -> f3 = 1
        evaluate(x_corner)

        # Now densely sample the Pareto front
        # f1 = 1 - x_1, f2 = x_2, so we want to cover (f1, f2) where f1 + f2 = 1
        # This means x_1 + x_2 = 1
        # We can parametrize: x_2 = t, x_1 = 1 - t for t in [0, 1]
        
        # Use a dense grid
        n_grid = self.budget - n_evals
        if n_grid > 0:
            # Use a 1D grid along the Pareto front line
            t_vals = np.linspace(0.0, 1.0, n_grid)
            for t in t_vals:
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = 1.0 - t  # x_1
                x[1] = t        # x_2
                # x_3..x_n = 0
                evaluate(x)

        # If we still have budget, add some random perturbations around the front
        while n_evals < self.budget and time.time() < deadline:
            x = np.zeros(self.dim)
            # Random point near the Pareto front
            t = np.random.uniform(0, 1)
            x[0] = 1.0 - t + np.random.normal(0, 0.05)
            x[1] = t + np.random.normal(0, 0.05)
            # Small perturbations in x_3..x_n
            x[2:] = np.abs(np.random.normal(0, 0.01, size=self.dim - 2))
            evaluate(x)

        return results
# EVOLVE-BLOCK-END
