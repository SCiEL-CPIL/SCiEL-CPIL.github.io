"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (rank-one + rank-mu) with deadline guard for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin under 300s / 330s limit

        rng = np.random.default_rng(12345)
        n = self.dim
        lam = 10  # population size
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff / 2 + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 / (n + 1.3) ** 2 + mu_eff / n) / (n + 4 + 2 / (n + 1.3) ** 2 + mu_eff / n)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / (n + 2) ** 2)

        # initial point: center of box
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # evaluate initial point
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1

        def _remaining():
            return self.budget - calls

        while calls < self.budget and time.time() < deadline:
            # sample population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                pop.append(y)

            # evaluate
            fvals = []
            for y in pop:
                if time.time() >= deadline:
                    break
                fy = func(y)
                fvals.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            # if we ran out of budget mid-generation, break
            if calls >= self.budget or time.time() >= deadline:
                break
            if len(fvals) < lam:
                break

            # sort
            order = np.argsort(fvals)
            sorted_y = np.array([pop[i] for i in order])
            sorted_f = np.array([fvals[i] for i in order])

            # new mean
            x_new = np.sum(w[:, None] * sorted_y[:mu], axis=0)

            # update C_inv_sqrt incrementally via Cholesky of C after update
            # First, recompute C update
            y_vec = (x_new - x_mean) / sigma
            # p_sigma update
            h_sigma = 1.0
            if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - self.budget // lam + 1)))
                    < (1 + 1.5 / n) * chi_n):
                h_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            # sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # p_c update
            h_c = 1.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

            # covariance update
            # rank one
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            # rank mu
            diff = (sorted_y[:mu] - x_mean) / sigma
            C += c_mu * np.sum(w[:, None, None] * np.outer(diff, diff).transpose(0, 2, 1), axis=0)

            x_mean = x_new

            # recompute C_inv_sqrt via Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # fallback: add small regularization
                C = C + 1e-6 * np.eye(n)
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)

            # early stopping: if sigma too small or no progress
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
