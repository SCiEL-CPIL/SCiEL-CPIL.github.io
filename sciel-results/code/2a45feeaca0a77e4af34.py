"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Deterministic Sobol quasi-random sequence for uniform initial coverage.
    2. Local Gaussian perturbations around non-dominated points for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol quasi-random initialization
        # Use a power of 2 for Sobol sampling
        n_sobol = 1
        while n_sobol < self.budget:
            n_sobol *= 2
        # Don't exceed budget
        n_sobol = min(n_sobol, self.budget)

        sobol = qmc.Sobol(d=self.dim, scramble=False)
        # Generate Sobol points in [0,1]^d
        points_unit = sobol.random(n_sobol)
        # Scale to bounds
        X = lo + points_unit * span
        F = np.array([problem(x) for x in X])
        used = n_sobol

        # Phase 2: Local refinement around non-dominated points
        # Find non-dominated points from the Sobol sweep
        def is_dominated(f, F_all):
            """Check if f is dominated by any point in F_all."""
            # A point dominates f if it is <= in all objectives and < in at least one
            better_or_equal = np.all(F_all <= f, axis=1)
            strictly_better = np.any(F_all < f, axis=1)
            return np.any(better_or_equal & strictly_better)

        nondom_mask = np.array([not is_dominated(f, F) for f in F])
        nondom_indices = np.where(nondom_mask)[0]

        if len(nondom_indices) == 0:
            # Fallback: use all points if no non-dominated found (shouldn't happen)
            nondom_indices = np.arange(len(F))

        # Use remaining budget for local perturbations
        remaining = self.budget - used

        if remaining > 0:
            rng = np.random.default_rng(42)
            # Start with a small step size, adapt if needed
            step_size = 0.05

            for i in range(remaining):
                # Select a non-dominated point to perturb
                idx = nondom_indices[rng.integers(len(nondom_indices))]
                parent = X[idx]

                # Gaussian perturbation
                perturbation = step_size * span * rng.standard_normal(self.dim)
                child = parent + perturbation
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace the worst point in the current set
                # Use a simple heuristic: replace the point with the largest sum of objectives
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = child
                F[worst_idx] = fc

                # Optionally update non-dominated set periodically
                if (i + 1) % max(1, remaining // 10) == 0:
                    nondom_mask = np.array([not is_dominated(f, F) for f in F])
                    nondom_indices = np.where(nondom_mask)[0]
                    if len(nondom_indices) == 0:
                        nondom_indices = np.arange(len(F))

        return X
# EVOLVE-BLOCK-END
