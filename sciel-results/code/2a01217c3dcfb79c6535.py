"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES with time guard for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # Adaptive CMA-ES implementation
        # Parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        # Step size control
        sigma = 1.0
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ / (dim + 1) - 1)) / (dim + 1)) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lambda_))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights @ weights)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        count_steps = 0
        best_x = x_mean.copy()
        best_f = func(best_x)

        # Precompute
        w = weights
        w_sum = w.sum()

        for gen in range(budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Use Cholesky or eigen decomposition for C_sqrt
                # For simplicity and speed in low dim, use eigen decomposition
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.maximum(evals, 1e-12)
                    C_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                except:
                    C_sqrt = np.eye(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate
            fitness = [func(y) for y in population]

            # Update best
            idx_min = np.argmin(fitness)
            if fitness[idx_min] < best_f:
                best_f = fitness[idx_min]
                best_x = population[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fitness)
            population = [population[i] for i in order]
            fitness = [fitness[i] for i in order]

            # Weighted mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * population[i]

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ y_w)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 100.0)

            # Update C
            if np.linalg.norm(p_sigma) < 1.4 * (1 + c_sigma) * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            rank_mu = np.outer(y_w, y_w)
            C = (1 + c_c * (2 - c_c) * h_sigma - c_1 - c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
                c_mu * rank_mu

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-12)
                C = evecs @ np.diag(evals) @ evecs.T
            except:
                C = np.eye(dim)

            # Update mean
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
