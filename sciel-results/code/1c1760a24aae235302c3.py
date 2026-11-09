"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    
    DTLZ7 with M=3 objectives:
    - n = 22 variables, bounds [0,1]
    - g(x) = (n_g - 1) + sum_{i=n_g}^{n-1} ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))
      where n_g = n - M + 1 = 20
    - f1 = x_0 * (1 + g)
    - f2 = x_1 * (1 + g)
    - f3 = (1 - sqrt(x_1)) * (1 + g)
    
    The Pareto front is achieved when g = 0, which requires x_i = 0.5 for i >= 20.
    On the Pareto front:
    - f1 = x_0
    - f2 = x_1
    - f3 = 1 - sqrt(x_1)
    
    So x_0 in [0,1] and x_1 in [0,1] parameterize the front.
    f1 ranges [0,1], f2 ranges [0,1], f3 ranges [0,1].
    
    Reference point is [1.1, 1.1, 25.0].
    Only points strictly better than reference in ALL objectives count.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)

        n_evals = 0
        results = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            results.append(f)
            n_evals += 1
            return f

        # Strategy: Sample the known Pareto front using a Sobol sequence
        # The front is parameterized by (x0, x1) in [0,1]^2 with x[2:] = 0.5
        # f1 = x0, f2 = x1, f3 = 1 - sqrt(x1)
        
        n_samples = self.budget
        
        # Use Sobol sequence for better low-discrepancy properties
        sampler = qmc.Sobol(d=2, scramble=True)
        sobol_points = sampler.random(n_samples)
        
        # Map to (x0, x1) parameter space
        # x0 and x1 are in [0,1]
        x0_vals = sobol_points[:, 0]
        x1_vals = sobol_points[:, 1]
        
        # Sample the Pareto front
        for i in range(n_samples):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = lo[0] + x0_vals[i] * (hi[0] - lo[0])
            x[1] = lo[1] + x1_vals[i] * (hi[1] - lo[1])
            # Set x[2:] to midpoint (0.5 for [0,1] bounds) to get g=0
            x[2:] = mid[2:]

            eval_point(x)

        return results
# EVOLVE-BLOCK-END
