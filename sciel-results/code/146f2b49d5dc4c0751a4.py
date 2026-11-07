"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        total_calls = 0

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of best individuals used for recombination
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chi-squared expectation
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * lambda_) + 1 / (21 * lambda_ ** 2))

        def cmaes(x_center, sigma, remaining_budget, deadline):
            nonlocal total_calls
            if remaining_budget <= 0 or time.time() >= deadline:
                return x_center, float('inf')

            x_best = x_center.copy()
            f_best = func(x_best)
            total_calls += 1

            mean = x_center.copy()
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            calls_used = 1

            # Precompute some values
            h_sigma = (1 + 2 / (dim + 1)) ** c_sigma
            # Update interval for C
            k = 0
            # We need to track the norm of p_sigma for h_sigma computation

            while calls_used < remaining_budget and time.time() < deadline:
                # Generate population
                Y = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim)) if False else C_sqrt
                    Y[i] = mean + sigma * (C_sqrt @ rng.standard_normal(dim))

                # Evaluate
                F = np.zeros(lambda_)
                for i in range(lambda_):
                    Y[i] = np.clip(Y[i], -5, 5)
                    F[i] = func(Y[i])
                    total_calls += 1
                    calls_used += 1
                    if total_calls >= budget or time.time() >= deadline:
                        break

                if calls_used >= remaining_budget or time.time() >= deadline:
                    break

                # Sort
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]

                # Update best
                if F[0] < f_best:
                    f_best = F[0]
                    x_best = Y[0].copy()

                # Weighted recombination
                y_w = np.sum(w[:, None] * Y[:mu], axis=0)

                # Update mean
                old_mean = mean.copy()
                mean = mean + c_c * (y_w - mean)

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (y_w / sigma)
                p_sigma_norm = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi_lambda else 0.0

                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (y_w / sigma)

                # Update C
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = C_inv_sqrt @ (Y[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(z_i, z_i)

                C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * (np.outer(mean - old_mean, mean - old_mean) + C) + c_mu * rank_mu

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                # Regularize
                C += 1e-10 * np.eye(dim)

                # Update C_inv and C_sqrt
                try:
                    C_inv = np.linalg.inv(C)
                    C_inv = (C_inv + C_inv.T) / 2
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_lambda - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check for convergence
                if sigma < 1e-8:
                    break

            return x_best, f_best

        rng = np.random.default_rng(42)

        # Start from center
        x0 = np.zeros(dim)
        f0 = func(x0)
        total_calls += 1

        # Try a few different starting points and step sizes
        starts = [
            (np.zeros(dim), 1.0),
            (np.zeros(dim), 0.5),
            (rng.uniform(-2, 2, dim), 1.0),
            (rng.uniform(-3, 3, dim), 1.5),
        ]

        best_x = x0.copy()
        best_f = f0

        # Allocate budget among restarts
        num_restarts = min(len(starts), 4)
        calls_per_restart = (budget - total_calls) // num_restarts

        for i in range(num_restarts):
            if total_calls >= budget or time.time() >= deadline:
                break
            x_start, sigma_start = starts[i]
            remaining = min(calls_per_restart, budget - total_calls)
            if remaining < 20:
                remaining = budget - total_calls
            x_res, f_res = cmaes(x_start, sigma_start, remaining, deadline)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
