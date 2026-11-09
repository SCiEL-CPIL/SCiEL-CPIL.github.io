"""Rewritten optimizer for BBOB-style f22 using a multi-start CMA-ES variant."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # For dim=5, standard CMA-ES settings
        mu = max(1, dim // 2)  # number of parents
        lambda_ = max(2 * mu, dim + 2)  # population size
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf

        # Multi-start: use a few initial points
        n_starts = 3
        calls_made = 0
        per_start_budget = budget // n_starts
        extra = budget % n_starts

        for start_idx in range(n_starts):
            if calls_made >= budget:
                break

            # Initial point: for f22 (Gallagher 21 peaks), the global optimum is known to be near specific locations
            # But since we don't know the hidden shift/rotation/offset, use diverse starts
            if start_idx == 0:
                x_mean = np.zeros(dim)
            elif start_idx == 1:
                x_mean = rng.uniform(-3, 3, dim)
            else:
                x_mean = rng.uniform(-4, 4, dim)

            # Initialize CMA-ES
            x_mean = np.clip(x_mean, -5, 5)
            sigma = 2.0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            step_size = np.ones(dim)

            # Evaluate initial point
            f_mean = func(x_mean)
            calls_made += 1
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            # CMA-ES loop
            budget_for_this_start = per_start_budget + (1 if start_idx < extra else 0)
            budget_for_this_start -= 1  # already used one call

            for gen in range(budget_for_this_start):
                if calls_made >= budget:
                    break

                # Sample population
                pop = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    pop[i] = x_mean + sigma * (step_size * rng.multivariate_normal(np.zeros(dim), C))
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate population
                f_pop = np.zeros(lambda_)
                for i in range(lambda_):
                    f_pop[i] = func(pop[i])
                    calls_made += 1
                    if f_pop[i] < best_f:
                        best_f = f_pop[i]
                        best_x = pop[i].copy()

                # Sort by fitness
                idx = np.argsort(f_pop)
                y = pop[idx]
                f_sorted = f_pop[idx]

                # Update mean
                x_new = np.sum(weights[:, None] * y[:mu], axis=0)

                # Update sigma (CSA)
                h_sigma = (
                    np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
                    < (1 + 2 / (dim + 1)) * chi_n
                )
                p_sigma = (
                    (1 - c_sigma) * p_sigma
                    + (h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma)
                    * (C @ (x_new - x_mean))
                )
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update correlation matrix
                p_c = (
                    (1 - c_c) * p_c
                    + (h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma)
                    * (x_new - x_mean)
                )
                C = (
                    (1 - c_1 - c_mu) * C
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim))
                    + c_mu * np.sum(weights[:, None, None] * np.outer(y[:mu] - x_mean, y[:mu] - x_mean), axis=0)
                )

                # Recompute step size
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                step_size = np.sqrt(eigvals)
                # Store eigenvectors for next iteration (simplified: just use C directly)

                x_mean = x_new
                f_mean = f_sorted[0]

        return best_x, best_f
# EVOLVE-BLOCK-END
