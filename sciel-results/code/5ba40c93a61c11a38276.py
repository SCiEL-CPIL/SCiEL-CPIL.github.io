"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Population size
        lam = 4
        mu = max(1, lam // 2)  # number of parents = 2

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Initial state
        x = np.zeros(dim)  # Start at origin
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        sigma = 1.0
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        calls_left = budget - 1
        generation = 0

        while calls_left >= lam:
            generation += 1
            # Generate lam offspring
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            offspring = []
            for i in range(lam):
                y = rng.standard_normal(dim)
                # Solve L v = y => v = C^{-1/2} y
                v = np.linalg.solve(L, y)
                x_new = x + sigma * v
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_left -= 1
                offspring.append((x_new, f_new))

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])

            # Select parents
            parents = [offspring[i][0] for i in range(mu)]

            # Update mean
            x_new = sum(weights[i] * parents[i] for i in range(mu))

            # Update step size
            diff = (x_new - x) / sigma
            try:
                v = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                v = diff

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * v

            # Hessian correction
            norm_p_sigma = np.sqrt(np.sum(p_sigma ** 2))
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * generation)) < threshold else 0.0

            sigma *= np.exp(c_sigma * d_sigma * (norm_p_sigma / chi_n - 1))

            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * c_c * v

            # Delta term
            delta_term = c_1 * (1 - (1 - c_c) ** (-2 * generation)) * np.outer(p_c, p_c)

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (parents[i] - x) / sigma
                try:
                    v_i = np.linalg.solve(L, diff_i)
                except np.linalg.LinAlgError:
                    v_i = diff_i
                rank_mu += weights[i] * np.outer(v_i, v_i)

            C = (1 - c_1 - c_mu) * C + c_1 * (delta_term + np.eye(dim)) + c_mu * rank_mu

            # Ensure C is symmetric and positive definite-ish
            C = (C + C.T) / 2

            # Update mean
            x = x_new

        return best_x, best_f

# EVOLVE-BLOCK-END
