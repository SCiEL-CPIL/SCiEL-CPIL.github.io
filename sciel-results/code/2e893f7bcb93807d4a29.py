"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using CMA-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for bi-modal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES with given initial point and budget."""
        n = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) /
                   ((n + 2) ** 2 + mu_eff))

        # Initialize
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        h_sigma = np.sqrt((1 - c_sigma) ** 2 * (n + 1) +
                          2 * c_sigma * (1 - c_sigma) * (n - 1))

        best_x = x.copy()
        best_f = fx

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lam):
                y = rng.standard_normal(n)
                ys.append(y)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                xi = np.array(xi, dtype=float)
                fi = func(xi)
                calls += 1
                xs.append(xi)
                fs.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

                if time.time() > deadline:
                    break

            if len(xs) < lam:
                break

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = np.array([xs[i] for i in order[:mu]])
            fs_sorted = [fs[i] for i in order[:mu]]

            # Update mean
            x_new = np.sum(weights[:, None] * xs_sorted, axis=0)

            # Update step size
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            if np.linalg.norm(p_sigma) / np.sqrt(
                    1 - (1 - c_sigma) ** (calls + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma *
                                (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(
                c_c * (2 - c_c) * mu_eff) * y_mean
            delta_h = (1 - c_c) ** (
                calls + 1) < 1 - 0.94 * (1 - c_c) ** (calls + 1)
            rank_c = np.outer(p_c, p_c)
            if not delta_h:
                rank_c += (1 - (1 - c_c) ** (calls + 1)) * np.eye(n)

            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = (xs_sorted[i] - x) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)

            C = (1 + c_1 * (rank_c - np.eye(n)) + c_mu * rank_mu) * C
            C += 1e-12 * np.eye(n)  # numerical stability

            # Recompute square root inverse
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-20, None)
            C_sqrt_inv = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

            x = x_new
            fx = best_f  # approximate

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # 20s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Initial sampling to get a good starting point
        n_init = min(50, self.budget // 10)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        remaining = self.budget - total_calls
        n_restarts = 5
        budget_per_restart = max(20, remaining // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if total_calls >= self.budget:
                break

            # Choose starting point: mix of best found and random
            if r == 0:
                x0 = best_x.copy()
            elif r % 2 == 1:
                # Perturb best
                x0 = best_x + 0.5 * rng.standard_normal(self.dim)
                x0 = np.clip(x0, -5, 5)
            else:
                # Fresh random
                x0 = rng.uniform(-5, 5, self.dim)

            sigma0 = 1.0 if r == 0 else 2.0

            x_opt, f_opt, calls_used = self._cma_es(
                func, x0, sigma0, min(budget_per_restart, self.budget - total_calls), rng, deadline)

            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
