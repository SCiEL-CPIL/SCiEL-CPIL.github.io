"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # 330s wall-clock limit per evaluation; use 315s as safety margin
        deadline = time.time() + 315.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Best solution found overall
        best_x = np.zeros(dim)
        best_f = np.inf

        # Multi-start strategy
        n_starts = 3
        budget_per_start = budget // n_starts

        for start in range(n_starts):
            if time.time() >= deadline:
                break

            # Initialize
            if start == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                if np.isfinite(best_f):
                    x = np.clip(best_x + rng.uniform(-3, 3, dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES-like parameters
            sigma = 1.0
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

            # Population size
            lambda_pop = 4 + int(3 * np.log(dim))
            remaining = budget - (start * budget_per_start)
            max_iter = min(remaining, budget_per_start)

            # State
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            mean = x.copy()
            f_mean = fx

            for it in range(max_iter):
                if time.time() >= deadline:
                    break

                # Generate population
                for i in range(lambda_pop):
                    if time.time() >= deadline:
                        break

                    # Sample from N(mean, sigma^2 * C)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)

                    z = rng.standard_normal(dim)
                    y = sigma * (L @ z)
                    trial_x = np.clip(mean + y, -5, 5)
                    trial_f = func(trial_x)

                    # Update best
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial_x.copy()

                    # Simple acceptance
                    if trial_f < f_mean:
                        mean = trial_x
                        f_mean = trial_f
                        sigma *= 1.03
                    else:
                        sigma *= 0.97

                    # Clamp sigma
                    sigma = np.clip(sigma, 0.01, 5.0)

                # Periodic adaptation
                if it > 0 and it % 20 == 0:
                    if sigma < 0.05:
                        mean = np.clip(mean + rng.uniform(-1, 1, dim), -5, 5)
                        sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
