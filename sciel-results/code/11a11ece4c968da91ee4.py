"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))

        # CMA-ES parameters
        sigma = 1.0  # initial step size
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 4)) / ((dim + 2) ** 2 + 5))
        cs = (dim + 1.7) / (dim + 9 + cs_factor) if False else (dim + 1.7) / (dim + 9)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # We'll maintain C^{-1/2} via Cholesky of C

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # We need to be careful with the update. Let's use a simpler but robust approach:
        # We'll maintain C and its Cholesky factor L such that C = L @ L.T
        # Sampling: y = L @ z, where z ~ N(0, I)
        # Actually standard CMA-ES: y = sigma * L @ z, x_new = x + y

        L = np.eye(dim)  # Cholesky factor: C = L @ L.T

        for gen in range(budget // lam):
            if time.time() > deadline:
                break

            # Generate population
            population = np.empty((lam, dim))
            fitness = np.empty(lam)

            for i in range(lam):
                z = rng.standard_normal(dim)
                y = sigma * (L @ z)
                x_new = np.clip(x + y, -5, 5)
                population[i] = x_new
                fitness[i] = func(x_new)

                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = x_new.copy()

            # Sort by fitness
            idx = np.argsort(fitness)
            fitness_sorted = fitness[idx]
            population_sorted = population[idx]

            # Weighted recombination
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            # New mean
            x_new_mean = np.sum(weights[:, None] * population_sorted[:mu], axis=0)

            # Update evolution paths
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_new_mean - x) / sigma
            # We need C^{-1/2}. Let's compute it from L: C = L @ L.T, so C^{-1/2} = (L^{-1})^T
            # Actually, C^{-1/2} @ v = (L^{-1})^T @ v if C = L L^T... let me be careful.
            # C = L @ L.T. C^{-1} = L^{-T} @ L^{-1}. C^{-1/2} is the matrix such that C^{-1/2} @ C^{-1/2} = C^{-1}.
            # One way: C^{-1/2} = (L^{-1})^T @ (L^{-1}) @ (L^{-1})^T ... no.
            # Simpler: compute C, then use scipy or manual inverse sqrt.
            # For dim=5, we can afford to compute C^{-1/2} each generation.

            C = L @ L.T
            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            delta_mean = (x_new_mean - x) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ delta_mean)
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n

            p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * delta_mean

            # Update covariance
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * np.outer(population_sorted[:mu] - x, population_sorted[:mu] - x), axis=0) / sigma ** 2

            C = (1 + mu_eff * cs * (2 - cs) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Re-cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x_new_mean

            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
