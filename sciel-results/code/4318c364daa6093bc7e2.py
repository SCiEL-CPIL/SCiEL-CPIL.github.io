"""Optimized program for BBOB f21 (Gallagher 101 peaks) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda = 4 + 3*ln(dim)
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x_best = np.zeros(dim)
        best_f = np.inf

        # Try a few initial points to find a good starting region
        for _ in range(min(10, budget)):
            x_try = rng.uniform(-4, 4, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                x_best = x_try.copy()

        x_mean = x_best.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        calls_used = min(10, budget)
        remaining = budget - calls_used

        for _ in range(remaining):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            B, D = np.linalg.eigh(C)
            D = np.sqrt(D)
            population = np.empty((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                population[i] = x_mean + sigma * (B * D) @ z

            # Evaluate population
            fitness = np.empty(pop_size)
            for i in range(pop_size):
                x_eval = np.clip(population[i], -5, 5)
                fitness[i] = func(x_eval)
                calls_used += 1
                if time.time() > deadline:
                    # Fill remaining with best known
                    fitness[i:] = fitness[:i].min()
                    break

            # Sort by fitness
            order = np.argsort(fitness)
            fitness_sorted = fitness[order]
            x_selected = population[order[:mu]]

            # Update best
            if fitness_sorted[0] < best_f:
                best_f = fitness_sorted[0]
                x_best = np.clip(x_selected[0], -5, 5).copy()

            # Update mean
            x_new_mean = (weights @ x_selected.T).T

            # Update evolution paths
            y = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # h_sigma: check if p_sigma is within expected range
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - min(10, budget) + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            # Update covariance matrix
            delta = h_sigma * c_sigma * (2 - h_sigma)
            C = (1 + c_1 + c_mu - c_1 - delta) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * sum(w * np.outer(x_selected[i] - x_mean, x_selected[i] - x_mean) for i, w in enumerate(weights))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 10.0)

            x_mean = x_new_mean

            # If we've used all budget, break
            if calls_used >= budget:
                break

        return x_best, best_f
# EVOLVE-BLOCK-END
