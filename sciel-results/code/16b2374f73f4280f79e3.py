"""Optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for ill-conditioned Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        x_best = None
        f_best = np.inf
        total_calls = 0

        def remaining():
            nonlocal total_calls
            return self.budget - total_calls

        def check_time():
            return time.time() >= deadline

        # CMA-ES parameters
        dim = self.dim
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 2.0
        # Initial mean: center of box
        mean = np.zeros(dim)

        # CMA-ES parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.3)) / ((dim + 1.6) * (dim + 0.5 - mu_eff / (dim + 1.3)))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) * (dim + mu_eff)))

        def cma_es(mean, sigma, budget_left):
            nonlocal f_best, x_best, total_calls

            if budget_left <= 0:
                return mean, sigma

            # Initialize covariance matrix
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)  # C^{-1/2}
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Precompute
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            while budget_left > 0 and not check_time():
                # Generate population
                X = np.empty((lambda_, dim))
                F = np.empty(lambda_)

                for i in range(lambda_):
                    if check_time():
                        break
                    z = np.random.randn(dim)
                    y = mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    X[i] = y
                    F[i] = func(y)
                    total_calls += 1
                    budget_left -= 1

                if total_calls >= self.budget or check_time():
                    break

                # Sort
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]

                # Update best
                if F[0] < f_best:
                    f_best = F[0]
                    x_best = X[0].copy()

                # Update mean
                mean_new = np.sum(weights[:, None] * X[:mu], axis=0)

                # Update step size
                y_mean = (mean_new - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget_left + 1)) / chi_n < 1.4 + 2 / (dim + 1))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                C = (1 + c_c * c_mu * (np.outer(p_c, p_c) - C) - c_c * C +
                     c_1 * (2 - c_1) * np.sum(weights, where=None) * 0 +
                     c_mu * np.sum(weights[:mu, None, None] * np.outer(X[:mu] - mean, X[:mu] - mean) / sigma ** 2, axis=0))
                # Simpler update:
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_c = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:mu, None, None] * np.outer(X[:mu] - mean, X[:mu] - mean), axis=0) / sigma ** 2
                C = (1 + c_c * delta_h - c_c - c_1) * C + c_1 * np.eye(dim) + c_mu * rank_mu

                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, 1e6)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                mean = mean_new
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))

                # Convergence check
                if np.linalg.norm(mean_new - mean) < 1e-10 and sigma < 1e-8:
                    break

            return mean, sigma

        # Multi-start CMA-ES
        n_starts = 3
        budget_per_start = self.budget // n_starts

        # First start: center
        mean0 = np.zeros(dim)
        f0 = func(mean0)
        total_calls += 1
        if f0 < f_best:
            f_best = f0
            x_best = mean0.copy()

        if not check_time() and total_calls < self.budget:
            m, s = cma_es(mean0.copy(), 2.0, self.budget - total_calls)

        # Additional starts if budget remains
        rng = np.random.default_rng(42)
        for i in range(1, n_starts):
            if check_time() or total_calls >= self.budget:
                break
            x_init = rng.uniform(-4, 4, dim)
            f_init = func(x_init)
            total_calls += 1
            if f_init < f_best:
                f_best = f_init
                x_best = x_init.copy()

            if not check_time() and total_calls < self.budget:
                m, s = cma_es(x_init.copy(), 1.5, self.budget - total_calls)

        if x_best is None:
            x_best = np.zeros(dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
