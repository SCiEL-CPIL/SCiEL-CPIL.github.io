"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Bent Cigar (f12) in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(d + 3))  # ~10 for d=5
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.dot(weights, weights)
        c_sigma = (mu_eff + 2) / (d + 4 + mu_eff)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2 / (d + 1.3)) / ((d + 1) ** 2 + mu_eff + 2)
        c_mu = min(1 - c_1, (2 + (mu_eff - 2) / d + 2 / (d + mu_eff - 2)) / ((d + 2) ** 2 + mu_eff + 2))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        hs = (1 + 0.34 + 2 / (d + 1)) * chi_n

        # Initialization
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        n_calls = 0
        f_best = np.inf
        x_best = x_mean.copy()

        def call_func(x):
            nonlocal n_calls, f_best, x_best
            n_calls += 1
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Initial evaluation
        call_func(x_mean)

        gen = 0
        while n_calls < budget:
            if time.time() > deadline:
                break

            # Generate and evaluate population
            xs = np.zeros((lambda_, d))
            fs = np.zeros(lambda_)

            for i in range(lambda_):
                if n_calls >= budget:
                    break
                z = np.random.randn(d)
                x_i = x_mean + sigma * (C_inv_sqrt @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                xs[i] = x_i
                fs[i] = call_func(x_i)

            if n_calls >= budget:
                break

            # Sort and select
            idx = np.argsort(fs)[:mu]
            xs_sel = xs[idx]
            fs_sel = fs[idx]

            # Weighted recombination
            x_new = np.sum(weights[:, None] * xs_sel, axis=0)

            # Update evolution paths
            y = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y)

            # Check h_sigma
            sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = (sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * hs

            # Update sigma
            if h_sigma:
                sigma *= np.exp(c_sigma / chi_n * (sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # Update C
            delta_h = (1 - (1 - c_c) ** (2 * (gen + 1))) / (1 - (1 - c_c) ** 2)
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                y_i = (xs_sel[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * (delta_h - 1) - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Periodically recompute C_inv_sqrt for numerical stability
            if (gen + 1) % 5 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-15, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    pass

            x_mean = x_new
            gen += 1

        return x_best, f_best
# EVOLVE-BLOCK-END
