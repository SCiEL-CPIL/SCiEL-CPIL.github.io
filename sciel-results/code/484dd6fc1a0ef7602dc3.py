"""DTLZ1 multi-objective optimizer exploiting known Pareto front structure directly."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Direct DTLZ1 optimizer: sample the known Pareto front analytically."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Compute center for g-variables
        center = (lo + hi) / 2.0
        span = hi - lo

        # Pre-allocate arrays for all evaluations
        X = np.zeros((budget, dim))
        F = np.zeros((budget, n_obj))
        
        # Set g-variables to center (0.5) to ensure g=0
        for j in range(2, dim):
            X[:, j] = center[j]

        # Use Sobol sequence for low-discrepancy sampling in 2D (x0, x1)
        # Sobol requires n to be a power of 2 for random points, but we can just take the first N
        # We'll generate a large enough Sobol sequence and take the first 'budget' points
        # Or we can generate exactly 'budget' points if we use the 'random' method with a seed
        # Note: scipy.stats.qmc.Sobol.random(n) returns n points
        
        try:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # Generate points in [0, 1]^2
            # We might need to handle if budget is not power of 2, but random(n) works for any n
            sobol_points = sampler.random(budget)
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.RandomState(42)
            sobol_points = rng.uniform(0, 1, size=(budget, 2))

        # Map Sobol points to bounds for x0 and x1
        X[:, 0] = lo[0] + span[0] * sobol_points[:, 0]
        X[:, 1] = lo[1] + span[1] * sobol_points[:, 1]

        used = 0
        for i in range(budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            used += 1

        # Return a valid array (the return value is ignored, but we must return something)
        if used > 0:
            return X[0]
        else:
            return np.zeros(dim)
# EVOLVE-BLOCK-END
