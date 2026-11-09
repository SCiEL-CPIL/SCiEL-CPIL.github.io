"""Rewritten optimizer using CMA-ES for BBOB-style f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 12  # Population size
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        sigma = 2.0
        mean = np.zeros(dim)
        C = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        evals_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        gen = 0
        while evals_used + lam <= budget:
            gen += 1
            # Generate population
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            z = rng.standard_normal((lam, dim))
            pop = mean + sigma * (z @ L.T)

            fitness = np.array([func(x) for x in pop])
            evals_used += lam

            idx_min = np.argmin(fitness)
            if fitness[idx_min] < best_f:
                best_f = fitness[idx_min]
                best_x = pop[idx_min].copy()

            order = np.argsort(fitness)
            y = (pop[order] - mean) / sigma

            mean = mean + sigma * np.sum(weights[:mu, np.newaxis] * y[:mu], axis=0)

            y_w = np.sum(weights[:mu, np.newaxis] * y[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sig = 1.0 if norm_p_sigma < (1 + 1.4 / dim) * chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) else 0.0

            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(y[:mu], y[:mu]), axis=0)

            C = (1 + c_1 * (h_sig * (2 - c_c) - 1) + c_mu) * C - c_1 * rank_one - c_mu * rank_mu
            C = (C + C.T) / 2

            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

        return best_x, best_f
# EVOLVE-BLOCK-END
