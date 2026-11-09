"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f17 (Schaffers F7) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2  # number of selected individuals

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial step size
        sigma = 2.0

        # Initial mean
        x_mean = rng.uniform(-5, 5, dim)

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # C_inv_sqrt = C^{-1/2}
        C_inv_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Best solution tracking
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        while calls_made < budget:
            if time.time() >= deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate population
            f_values = []
            for y in population:
                fy = func(y)
                calls_made += 1
                f_values.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort by fitness
            indices = np.argsort(f_values)
            population_sorted = [population[i] for i in indices]

            # Compute new mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population_sorted[i]

            # Update evolution path for sigma
            diff = x_new - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma

            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma

            # Update covariance matrix
            H = np.zeros((dim, dim))
            for i in range(mu):
                d = population_sorted[i] - x_mean
                H += weights[i] * np.outer(d, d)
            H = H / (sigma ** 2)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(dim)) + c_mu * H
            C = (C + C.T) / 2

            # Compute C^{-1/2} using eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C_inv_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Update mean
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
