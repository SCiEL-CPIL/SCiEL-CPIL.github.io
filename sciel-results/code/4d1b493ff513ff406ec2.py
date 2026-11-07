"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Global Optimization using Sobol sequence with adaptive local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol global exploration
        # Use Sobol sequence for low-discrepancy sampling
        # Reserve some budget for local refinement
        n_sobol = int(budget * 0.7)  # 70% for global
        n_local = budget - n_sobol   # 30% for local refinement

        # Generate Sobol samples
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)

        # Map to [-5, 5]^dim
        sobol_mapped = sobol_samples * 10.0 - 5.0

        # Evaluate all Sobol points
        best_x = None
        best_f = np.inf
        all_points = []
        all_values = []

        for i in range(n_sobol):
            x = sobol_mapped[i]
            fx = func(x)
            all_points.append(x)
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement around the best point found
        # Use a small population-based local search (CMA-like or simple ES)
        # Start with a small neighborhood around best_x

        if n_local > 0:
            # Adaptive local search using a simple (1+lambda)-ES with shrinking sigma
            sigma = 1.0  # initial step size for local search
            x_best = best_x.copy()
            f_best = best_f

            # Use a small population for local refinement
            pop_size = min(5, n_local // 2) if n_local >= 10 else 1
            remaining = n_local

            while remaining > 0:
                # Generate offspring
                for _ in range(min(pop_size, remaining)):
                    if remaining <= 0:
                        break
                    # Gaussian perturbation
                    perturbation = sigma * np.random.randn(dim)
                    y = np.clip(x_best + perturbation, -5, 5)
                    fy = func(y)
                    remaining -= 1

                    if fy < f_best:
                        x_best = y.copy()
                        f_best = fy
                        # Increase sigma slightly on success
                        sigma *= 1.2
                    else:
                        # Decrease sigma on failure
                        sigma *= 0.8

                # If sigma becomes too small, stop
                if sigma < 0.01:
                    break

            best_x = x_best
            best_f = f_best

        return best_x, best_f
# EVOLVE-BLOCK-END
