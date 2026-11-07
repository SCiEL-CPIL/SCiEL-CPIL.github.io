"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard CMA-ES settings
        lambda_pop = 4 * int(np.floor(3 + np.log(dim)))  # population size
        mu = lambda_pop // 2  # number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point and step size
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0

        # Covariance matrix and its decomposition
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best solution
        best_x = x.copy()
        best_f = func(x)

        calls_made = 1

        for iteration in range(budget - 1):
            if calls_made >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if calls_made >= budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            y_sorted = [offspring[i] for i in sorted_idx]
            f_sorted = [offspring_f[i] for i in sorted_idx]

            # Compute weighted mean
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * y_sorted[i]

            # Update evolution paths
            d = m - x
            x_sigma = d / (sigma * chi_n) if sigma > 0 else np.zeros(dim)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * x_sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d / sigma

            # Update mean
            x = m

            # Update covariance matrix
            delta_diag = 1.0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                delta_diag = 1 + c_sigma * (2 - c_sigma)

            C = (1 + c_1 * delta_diag - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * sum(weights[i] * np.outer(y_sorted[i] - x, y_sorted[i] - x) for i in range(mu))

            # Ensure positive definiteness
            C = 0.5 * (C + C.T)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)
            except np.linalg.LinAlgError:
                C = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update matrix square roots (approximate via Cholesky-like update)
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
