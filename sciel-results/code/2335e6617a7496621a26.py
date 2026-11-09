"""Optimized program: CMA-ES for BBOB Rosenbrock (f8)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        time_limit = 320.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4 + int(3 * np.log(dim)), 2 * dim)
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - cc + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)

        # Evaluate initial point
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Eigendecomposition for sampling
        B, D = np.linalg.eigh(C)
        D = np.sqrt(np.clip(D, 1e-20, None))

        for iteration in range(budget - 1):
            if time.time() - t_start > time_limit:
                break
            if calls >= budget:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            offspring_f = np.full(lam, np.inf)
            n_eval_this_iter = 0

            for i in range(lam):
                if calls >= budget or time.time() - t_start > time_limit:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (z / D))
                y = np.clip(y, -5, 5)
                offspring[i] = y
                offspring_f[i] = func(y)
                calls += 1
                n_eval_this_iter += 1

            if n_eval_this_iter == 0:
                break

            # Sort offspring by fitness
            idx = np.argsort(offspring_f[:n_eval_this_iter])
            y_sorted = offspring[idx[:min(mu, n_eval_this_iter)]]
            f_sorted = offspring_f[idx[:min(mu, n_eval_this_iter)]]
            mu_actual = min(mu, n_eval_this_iter)

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean
            w_use = w[:mu_actual]
            w_use = w_use / w_use.sum()
            m = np.sum(w_use[:, None] * y_sorted, axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * (B.T @ (m - x) / sigma)
            ps_norm = np.linalg.norm(ps)
            hsig = ps_norm / (np.sqrt(1 - (1 - cs) ** (iteration + 1)) * chi_n) < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - x) / sigma
            delta_hsig = (1 - hsig) * cc * (1 - cc)

            # Update covariance matrix
            rank_mu = np.sum(w_use[:, None] * (y_sorted - m)[:, None] * (y_sorted - m)[None, :], axis=0)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * delta_hsig * C) + cmu * rank_mu
            C = (C + C.T) / 2  # symmetrize

            # Update step size (CSA)
            sigma *= np.exp(cs / damps * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update mean
            x = m

            # Re-orthogonalize periodically
            if iteration % 50 == 0 or iteration == 0:
                try:
                    B, D_new = np.linalg.eigh(C)
                    D = np.sqrt(np.clip(D_new, 1e-20, None))
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
                    D = np.ones(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
