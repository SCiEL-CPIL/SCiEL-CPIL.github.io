"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the rotated Rosenbrock (f9) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Initial mean and sigma
        x_mean = rng.normal(0, 0.5, n)
        sigma = 0.5

        # Covariance matrix
        C = np.eye(n)
        C_sqrt = np.eye(n)

        # Adaptation rates
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + cs
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Best tracking
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        # Evaluate initial point
        fx_init = func(x_mean)
        calls += 1
        best_f = fx_init
        best_x = x_mean.copy()

        # Main loop
        while calls + lam <= budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = x_mean + sigma * (C_sqrt @ z)

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Update sigma path
            d_norm = np.linalg.norm(np.linalg.solve(C_sqrt.T, (x_mean - x_old) / sigma))
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.solve(C_sqrt.T, (x_mean - x_old) / sigma)

            # Update C path
            h_sigma = d_norm < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c) + delta_h * np.eye(n)
            diff = pop_sorted[:mu] - x_old
            rank_mu = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            # Ensure PD
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(n)
                C = (C + C.T) / 2
                C_sqrt = np.linalg.cholesky(C)

            # Update sigma
            sigma *= np.exp(cs / ds * (d_norm / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
