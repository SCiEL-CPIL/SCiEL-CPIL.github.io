"""Rewritten optimizer for BBOB-style Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES style optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Weierstrass is multimodal; use multiple restarts with CMA-ES-like population
        # Budget 500, dim 5. Use population size ~ 4+2*floor(3*ln(5)) ≈ 4+2*5 = 14
        # But that's expensive. Let's use a smaller population for more restarts.
        # Actually, with budget 500 and dim 5, let's use population size 8.
        # Number of full generations per restart: ~62. Let's do 4 restarts of ~125 calls each.

        lambda_pop = 8
        max_restarts = 4
        calls_per_restart = budget // max_restarts  # 125

        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize population around best found so far or random
            if best_x is not None and restart > 0:
                # Start near best with some spread
                sigma_init = 1.0 / (restart + 1)
                x_center = best_x.copy()
            else:
                x_center = rng.uniform(-5, 5, dim)
                sigma_init = 2.0

            # Population CMA-ES
            mu = lambda_pop // 2  # number of parents
            # Weights
            weights = np.ones(mu)
            weights[0] = np.log(mu + 0.5) - np.log(mu // 2)
            weights /= weights.sum()
            mueff = 1.0 / (weights ** 2).sum()

            # Step size control
            sigma = sigma_init
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

            # Covariance matrix
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            pop_size = lambda_pop
            remaining = min(calls_per_restart, budget - calls_made)
            max_gens = remaining // pop_size

            for gen in range(max_gens):
                if calls_made + pop_size > budget:
                    break
                if time.time() > deadline:
                    break

                # Sample population
                D = np.sqrt(np.diag(C)) if not np.allclose(C, np.eye(dim)) else np.ones(dim)
                B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                B_inv = np.linalg.inv(B)
                B_inv_T = B_inv.T

                z = rng.standard_normal((pop_size, dim))
                y = x_center + sigma * (z @ B.T)
                # Clip to bounds
                y = np.clip(y, -5, 5)

                # Evaluate
                fitness = np.empty(pop_size)
                for i in range(pop_size):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fitness[i] = func(y[i])
                    calls_made += 1
                    if fitness[i] < best_f:
                        best_f = fitness[i]
                        best_x = y[i].copy()

                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fitness)
                y_sorted = y[order]
                fitness_sorted = fitness[order]

                # Update best
                if fitness_sorted[0] < best_f:
                    best_f = fitness_sorted[0]
                    best_x = y_sorted[0].copy()

                # Mean
                m_new = y_sorted[:mu] @ weights[:, None]

                # Update step size
                y_mean = (m_new - x_center) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * B_inv_T @ y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update evolution path
                pc = (1 - c_c) * pc + c_c * (B_inv_T @ y_mean)

                # Update covariance
                delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) * c_c
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = B_inv_T @ ((y_sorted[i] - x_center) / sigma)
                    rank_mu += weights[i] * np.outer(yi, yi)

                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Symmetrize
                C = (C + C.T) / 2

                x_center = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
