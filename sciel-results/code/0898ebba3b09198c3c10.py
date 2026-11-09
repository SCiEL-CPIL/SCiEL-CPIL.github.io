"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * mu     # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-3, 3, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)

        calls = 1

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop = np.empty((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                pop[i] = x + sigma * (C_sqrt @ z)

            # Evaluate
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                xi = np.clip(pop[i], -5, 5)
                fi = func(xi)
                fvals[i] = fi
                calls += 1
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()

            if calls >= budget or time.time() > deadline:
                break

            # Sort
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Weighted mean
            m = np.zeros(n)
            for i in range(mu):
                m += weights[i] * pop_sorted[i]

            # Update step-size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt @ (m - x))
            norm_ps = np.linalg.norm(ps)

            # Update covariance adaptation
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * c_c * (m - x) / sigma

            delta_hsig = (1 - hsig) * c_c * (1 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            for i in range(mu):
                diff = pop_sorted[i] - x
                C += (c_1 * delta_hsig + c_mu * weights[i]) * np.outer(diff, diff)

            # Update mean
            x = m

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Update C_sqrt (Cholesky)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(n)
                C = np.eye(n)

            # Update current best if improved
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = pop_sorted[0].copy()
                best_fx = min(best_fx, fx)
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
