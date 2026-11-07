"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        def evaluate(x):
            self.calls += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        def cma_es(x0, sigma, budget_left):
            """Run CMA-ES starting from x0 with initial sigma."""
            n = self.dim
            lam = 4 + int(3 * math.log(n))
            lam = max(lam, n + 1)
            mu = lam // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

            x_mean = x0.copy()
            C = np.eye(n)
            C_sqrt = np.eye(n)
            step_size = sigma
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n**2))

            gen = 0
            while self.calls < self.budget and budget_left > 0:
                if time.time() >= deadline:
                    break

                # Sample population
                z = np.random.randn(lam, n)
                pop = x_mean + step_size * (z @ C_sqrt)
                pop = np.clip(pop, -5, 5)

                # Evaluate
                fvals = np.array([evaluate(x) for x in pop])
                budget_left -= lam

                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_new_mean = np.sum(weights[:mu, None] * pop[:mu], axis=0) / np.sum(weights[:mu])

                # Update paths
                ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff) / step_size * (C_sqrt.T @ (x_new_mean - x_mean))
                norm_ps = np.linalg.norm(ps)
                hsig = 1.0 if norm_ps / math.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

                pc = (1 - c_c) * pc + hsig * math.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / step_size

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank1 = np.outer(pc, pc)
                delta_vec = x_new_mean - x_mean
                rankmu = np.sum(weights[:mu, None] * (pop[:mu] - x_mean)[:, None] * (pop[:mu] - x_mean)[None, :], axis=0)
                rankmu = rankmu / np.sum(weights[:mu])

                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                C = (C + C.T) / 2 + 1e-12 * np.eye(n)

                try:
                    C_sqrt = np.linalg.cholesky(C).T
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)

                # CSA
                step_size *= math.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

                x_mean = x_new_mean
                gen += 1

                if step_size < 1e-15:
                    break

            return x_mean, step_size

        # Multiple restarts
        n_restarts = 4
        budget_per = self.budget // n_restarts

        for r in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() >= deadline:
                break

            if r == 0:
                x0 = np.random.uniform(-5, 5, self.dim)
                sigma = 2.0
            else:
                x0 = self.best_x + np.random.normal(0, 0.5, self.dim)
                x0 = np.clip(x0, -5, 5)
                sigma = 1.0

            remaining = self.budget - self.calls
            cma_es(x0, sigma, min(budget_per, remaining))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
