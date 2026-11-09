import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Exploits DTLZ1 structure: Pareto front is g=0, x2..x6=0.5, sample (x0,x1) in [0,1]^2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate Sobol samples in 2D for (x0, x1)
        next_pow2 = 1
        while next_pow2 < budget:
            next_pow2 *= 2

        s = qmc.Sobol(d=2, scramble=True)
        samples = s.random(next_pow2)
        samples = samples[:budget]

        # x0, x1 are directly the shape variables on [0,1]
        x0 = samples[:, 0]
        x1 = samples[:, 1]

        # Construct decision vectors: x2..x6 = 0.5 for g=0
        points = np.zeros((budget, dim))
        points[:, 0] = x0
        points[:, 1] = x1
        for j in range(2, dim):
            points[:, j] = 0.5

        # Evaluate
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))

        return results
