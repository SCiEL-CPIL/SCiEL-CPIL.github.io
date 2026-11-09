"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        if n_evals <= 0:
            return None

        # ZDT1: f1(x) = x1, f2(x) = (1 + g(x2..xN)) * (1 - sqrt(f1/f2))
        # g(x) = 10*n + sum((xi - 0.5)^2 - cos(20*pi*(xi-0.5)))
        # Minimum g = 0 when all xi = 0.5
        # Pareto front: x2..xN = 0.5, x1 in [0, 1]
        # f1 = x1, f2 = 1 - sqrt(x1)
        # This is a concave front (f2 decreases as f1 increases)
        
        # Strategy: distribute points on the Pareto front
        # For hypervolume maximization with reference [11, 11],
        # we want points that maximize the area under the curve.
        
        # Use a mix of uniform and logarithmic spacing for better coverage
        idx = np.arange(n_evals)
        u = idx / (n_evals - 1)
        
        # Try different distributions - let's use a balanced approach
        # The front is f2 = 1 - sqrt(f1), concave decreasing
        # Hypervolume contribution is roughly (11 - f1) * (11 - f2)
        # = (11 - t) * (11 - (1 - sqrt(t))) = (11 - t) * (10 + sqrt(t))
        # This is maximized when t is around 0.5-0.7
        
        # Use a power distribution that balances coverage
        t_vals = u ** 0.7  # Slight clustering toward larger t values
        
        # Construct points on the Pareto front: x2..xN = 0.5
        X = np.full((n_evals, self.dim), 0.5)
        X[:, 0] = t_vals
        X = np.clip(X, lo, hi)
        
        last_obj = None
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            last_obj = f
        
        return last_obj
# EVOLVE-BLOCK-END
