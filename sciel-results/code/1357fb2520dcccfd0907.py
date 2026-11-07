"""Optimizer for BBOB f19 (Griewank-Rosenbrock) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES for a given budget, respecting deadline."""
        n = self.dim
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))  # population size
        c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w @ w)
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # State
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        calls = 0
        best_x = x.copy()
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(n)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial population
        for _ in range(lambda_):
            y = x + sigma * rng.standard_normal(n)
            safe_call(y)

        if calls >= budget:
            return best_x, best_f

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, n))
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                ys[i] = x + sigma * rng.standard_normal(n)
                fs[i] = safe_call(ys[i])

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Update mean
            y_w = (w[:, None] * ys[:mu]) @ (1.0)  # weighted sum
            x_old = x.copy()
            x = x + (sigma / np.sqrt(c_1 * mu_eff)) * (w @ ys[:mu] - x + x)  # simplified
            x = x_old + (w @ ys[:mu] - x_old)

            # Update sigma (CSA)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (ys[:mu].T @ w / np.sqrt(mu_eff) - x))
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance (rank-1 and rank-mu updates)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_ + 1)))) < (1 + 4 / (n + 1)) * np.sqrt(c_sigma * (2 - c_sigma))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma
            v_sigma = (1 - (1 - c_sigma) ** 2) * np.outer(p_sigma, p_sigma)
            v_c = (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c)
            C = (1 + c_1 * mu_eff * (h_sigma - 1) - c_1 - c_mu) * C + c_1 * np.outer(x - x_old, x - x_old) / sigma ** 2 + c_mu / sigma ** 2 * (w @ (ys[:mu] - x[:, None]).T) @ (ys[:mu] - x[:, None])

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Re-orthogonalize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        deadline = time.time() + 290.0  # safety margin before 300s kill

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Stage 1: Brief coarse exploration (10 points) to find a good starting basin
        n_coarse = min(10, budget)
        for _ in range(n_coarse):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_call(x)

        if calls >= budget:
            return best_x, best_f

        # Stage 2: CMA-ES from best point
        x0 = best_x.copy()
        sigma0 = 1.0
        remaining = budget - calls

        best_x, best_f = self._cma_es(func, x0, sigma0, remaining, deadline, rng)

        # Update global best
        if best_f < np.inf:
            pass  # already tracked inside _cma_es via safe_call

        return best_x, best_f
# EVOLVE-BLOCK-END
