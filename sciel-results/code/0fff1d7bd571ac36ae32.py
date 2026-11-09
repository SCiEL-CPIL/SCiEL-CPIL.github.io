"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Global best
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with a few random points
        rng = np.random.default_rng(42)
        for _ in range(min(5, budget)):
            x0 = rng.uniform(-5, 5, dim)
            evaluate(x0)

        remaining = budget - min(5, budget)

        # CMA-ES parameters
        lambda_ = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Run CMA-ES with restarts
        while remaining > 0 and time.time() < deadline:
            # Start from best found so far
            x = best_x.copy()
            sigma = 2.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Local budget: use up to 100 evaluations per restart
            local_budget = min(remaining, 100)
            local_calls = 0

            for gen in range(local_budget):
                if time.time() >= deadline:
                    break

                # Generate population
                pops = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * z
                    pops.append(y)

                # Evaluate
                fitness = np.array([evaluate(y) for y in pops])
                local_calls += lambda_
                remaining -= local_calls

                # Sort
                order = np.argsort(fitness)
                selected = [pops[i] for i in order[:mu]]
                selected_f = fitness[order[:mu]]

                # Update mean
                x_new = np.sum(weights[:, None] * np.array(selected), axis=0)

                # Update covariance
                y_vecs = np.array(selected) - x
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(y_vecs.mean(axis=0), y_vecs.mean(axis=0))
                for i, w_i in enumerate(weights):
                    C += w_i * np.outer(y_vecs[i], y_vecs[i])

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.norm(x_new - x) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update evolution paths
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                x = x_new

            if remaining <= 0 or time.time() >= deadline:
                break

            # If no improvement, try a different starting point
            if local_calls == 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
