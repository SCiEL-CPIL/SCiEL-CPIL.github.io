"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with adaptive covariance for multimodal search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        try:
            # CMA-ES parameters
            pop_size = 12
            mu = pop_size // 2  # number of selected parents
            # Weights for selected parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = weights.sum() ** 2 / (weights ** 2).sum()
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initialize
            x_mean = best_x.copy() if calls_made > 0 else rng.uniform(-4.0, 4.0, dim)
            # If we haven't evaluated yet, do a quick initial scan
            if calls_made == 0:
                x_mean = rng.uniform(-4.0, 4.0, dim)
                fx = evaluate(x_mean)
                best_x = x_mean.copy()
                best_f = fx

            sigma = 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            eps = 1e-12

            def mat_sqrt_inv(C):
                """Compute C^{-1/2} using eigen decomposition."""
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-15)
                inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
                return inv_sqrt

            # Main CMA-ES loop
            while calls_made < budget and time.time() < deadline:
                # Generate population
                C_inv_sqrt = mat_sqrt_inv(C)
                population = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (C_inv_sqrt @ z)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    population.append(x_new)

                # Evaluate population
                fitness = []
                for x in population:
                    fx = evaluate(x)
                    fitness.append(fx)
                    if calls_made >= budget or time.time() >= deadline:
                        break

                if len(fitness) < pop_size:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(fitness)
                selected = [population[i] for i in sorted_idx[:mu]]
                selected_fitness = [fitness[i] for i in sorted_idx[:mu]]

                # Update best
                if selected_fitness[0] < best_f:
                    best_f = selected_fitness[0]
                    best_x = selected[0].copy()

                # Update mean
                x_old = x_mean.copy()
                x_mean = sum(w * x for w, x in zip(weights, selected))

                # Update sigma (CSA)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ (x_mean - x_old))
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if norm_p_sigma / chi_n < (1 + 2 / (dim + 1)) * (1 - c_sigma) else 0.0
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-6, 10.0)

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_old) / sigma
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(w * np.outer(x - x_old, x - x_old) for w, x in zip(weights, selected)) / sigma ** 2
                C = (1 + c_1 * (h_sigma * (dim + 1) - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

                # Check for convergence
                if np.max(eigenvalues) / np.min(eigenvalues) > 1e12 or sigma < 1e-8:
                    # Restart with fresh exploration
                    if calls_made < budget - 20:
                        x_mean = best_x + rng.normal(0, 1.0, dim)
                        x_mean = np.clip(x_mean, -5.0, 5.0)
                        sigma = 0.5
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                    else:
                        break

        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
