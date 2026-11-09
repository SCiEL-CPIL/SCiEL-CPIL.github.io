"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size for CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Initial point: try center first, then random if needed
        x0 = np.zeros(dim)
        f0 = func(x0)

        # If center is bad, try a few random starts
        best_x = x0.copy()
        best_f = f0
        for i in range(5):
            x_try = rng.uniform(-4, 4, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        x = best_x.copy()
        f_best = best_f
        calls_used = 6  # 1 center + 5 random

        # CMA-ES parameters
        sigma = 1.0
        c1 = (2 + 2 / (dim + 1.7)) / ((dim + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1.7)) / ((dim + 1) ** 2 + 2))
        cs = (dim + 1.7 + 2 / (dim + 1.7)) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        mu_eff = mu * (1 - 1 / mu) if mu > 1 else 1.0

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        C_eigvals = np.clip(C_eigvals, 1e-12, 1e12)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        for iteration in range(budget - calls_used):
            if calls_used >= budget:
                break

            # Sample population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y_clipped = np.clip(y, -5, 5)
                offspring.append(y_clipped)
                calls_used += 1

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
                if fy < f_best:
                    f_best = fy
                    best_x = y.copy()

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices[:mu]]
            sorted_f = [offspring_f[i] for i in indices[:mu]]

            # Update best
            if sorted_f[0] < f_best:
                f_best = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Mean update
            w_mean = np.sum(weights)
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)

            # Check if we should update the mean
            if np.linalg.norm(x_new - x) > 0:
                # Success rule: if improvement, increase sigma
                improvement = f_best - np.mean(sorted_f)
                if improvement > 0:
                    sigma *= 1.1
                else:
                    sigma *= 0.95

                # Update evolution paths
                z_mean = C_inv_sqrt @ (x_new - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) / \
                       np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) * (1 - cs)
                hsig = min(hsig, 1.4 + 2 / (dim + 1))

                x = x_new
                if hsig < 1.4 + 2 / (dim + 1):
                    pc = (1 - c1) * pc + c1 * z_mean
                    C = (1 - c1 * (2 - c1)) * C + c1 * (2 - c1) * np.outer(pc, pc)

                # Rank-mu update
                z_sorted = [C_inv_sqrt @ (sorted_offspring[i] - x_old) / sigma for i in range(mu)]
                C = (1 - cmu * (2 - c1 - cmu)) * C + cmu * np.sum(
                    [weights[i] ** 2 * np.outer(z_sorted[i], z_sorted[i]) for i in range(mu)], axis=0)

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals))

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-8, 10)

        return best_x, f_best
# EVOLVE-BLOCK-END
