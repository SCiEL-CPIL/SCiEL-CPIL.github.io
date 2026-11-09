"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1

        while evals < budget:
            # Generate population using Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)

            offspring = []
            offspring_f = []
            for i in range(lam):
                z = np.random.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)
                if evals >= budget:
                    break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring_sorted = offspring[order]
            offspring_f_sorted = offspring_f[order]

            # Update best
            if offspring_f_sorted[0] < best_f:
                best_f = offspring_f_sorted[0]
                best_x = offspring_sorted[0].copy()

            # Save old mean
            x_mean_old = x_mean.copy()

            # Update mean
            x_mean = x_mean_old + sum(weights[i] * (offspring_sorted[i] - x_mean_old) for i in range(mu))

            # Update path accumulators
            y_vec = (x_mean - x_mean_old) / sigma if sigma > 1e-15 else np.zeros(n)
            pc = (1 - c_c) * pc + c_c * y_vec
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * y_vec

            # Step size control
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e15)

            # Covariance matrix update
            C = (1 + c_1 * (2 - c_c) - c_mu) * C
            C += c_1 * np.outer(pc, pc)
            for i in range(mu):
                diff = offspring_sorted[i] - x_mean_old
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = (eigvecs * eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
