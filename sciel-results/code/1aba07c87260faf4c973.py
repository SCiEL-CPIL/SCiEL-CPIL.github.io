"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES from x0 with initial sigma sigma0, using at most budget calls."""
        n = self.dim
        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lam = 2 * mu  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1.7) + c_sigma) / ((n + 2) * (1 - c_sigma))
        c_mu = (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 * (1 - c_1))
        c_mu = max(0, min(c_mu, 1 - c_1))

        # Initialize
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls_made = 0
        best_x = x.copy()
        best_fx = func(x)
        calls_made += 1

        for gen in range(budget // lam + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            A = np.linalg.cholesky(C)
            ys = []
            fs = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * A @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)

            if len(ys) < mu:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx[:mu]]
            fs = [fs[i] for i in idx[:mu]]

            # Update best
            if fs[0] < best_fx:
                best_fx = fs[0]
                best_x = ys[0].copy()

            # Compute mean of selected points
            y_mean = sum(weights[i] * ys[i] for i in range(mu))

            # Update evolution paths
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (y_mean - x) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 + (1 - c_sigma) ** (2 * (gen + 1)))
            pc = (1 - c_c) * pc + hsig * c_c * (y_mean - x) / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig) * C
            C += c_1 * np.outer(pc, pc)
            C += c_mu * sum(weights[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))

            # Update step size
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = y_mean

        return best_x, best_fx, calls_made

    def __call__(self, func):
        # Safety margin: 280s is well within the 330s kill limit.
        deadline = time.time() + 280.0
        rng = np.random.default_rng(0)

        # Start from center of the box
        x0 = np.zeros(self.dim)
        sigma0 = 2.0

        best_x, best_fx, calls_used = self._cma_es(func, x0, sigma0, self.budget, deadline, rng)

        return best_x, best_fx
# EVOLVE-BLOCK-END
