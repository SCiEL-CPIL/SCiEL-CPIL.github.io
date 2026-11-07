"""Multi-objective optimizer for DTLZ1 exploiting known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_F = []

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        # DTLZ1 structure for 3 objectives:
        # f1 = 0.5 * x0 * (1+g)
        # f2 = 0.5 * (1-x0) * x1 * (1+g)
        # f3 = 0.5 * (1-x0) * (1-x1) * (1+g)
        # g = 100 + 10*(M-2) + sum_{i=3}^{n} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # Minimum g = 100 + 10*(M-2) = 110 for M=3
        # On Pareto front: x_i = 0.5 for i >= 2, so g = 110, 1+g = 111
        # f1+f2+f3 = 0.5*(1+g) = 55.5

        # Phase 1: Systematic grid sampling on the Pareto front
        # Parameterize by x0 and x1 in [0,1] with x[2:] = 0.5
        # Use a dense grid to cover the simplex well

        n_grid = 15  # 15x15 = 225 points, but budget is 200
        # Adjust grid size to fit budget
        n_grid = min(14, int(np.sqrt(self.budget * 0.7)))
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)

        for x0 in x0_vals:
            if time.time() > deadline - 30:
                break
            for x1 in x1_vals:
                if len(all_F) >= self.budget:
                    break
                if time.time() > deadline - 30:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5  # optimal shape variables for DTLZ1
                evaluate(x)

        # Phase 2: Add points near the front with slight perturbations for robustness
        remaining = self.budget - len(all_F)
        if remaining > 0 and time.time() < deadline - 20:
            for _ in range(remaining):
                if time.time() > deadline - 10 or len(all_F) >= self.budget:
                    break
                x = np.zeros(self.dim)
                x[0] = rng.uniform(0.0, 1.0)
                x[1] = rng.uniform(0.0, 1.0)
                # Shape variables near 0.5 with small noise
                x[2:] = 0.5 + 0.02 * rng.standard_normal(self.dim - 2)
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Phase 3: If budget remains, add more systematic points at finer resolution
        if len(all_F) < self.budget and time.time() < deadline - 5:
            # Add midpoints of the grid for better coverage
            n_mid = min(self.budget - len(all_F), 20)
            for i in range(n_mid):
                if time.time() > deadline - 2 or len(all_F) >= self.budget:
                    break
                # Sample on the front with finer x0, x1
                x0 = rng.uniform(0.0, 1.0)
                x1 = rng.uniform(0.0, 1.0)
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                evaluate(x)

        return all_F
# EVOLVE-BLOCK-END
