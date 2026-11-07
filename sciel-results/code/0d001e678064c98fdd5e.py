"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style covariance matrix adaptation optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Population size
        lam = 4 + int(3 * np.log(n))  # standard CMA-ES lambda
        mu = lam // 2

        # Initial parameters
        x0 = np.zeros(n)
        sigma0 = 2.0
        c1 = (2 + 2 / (n + 1)) / ((n + 1.3) ** 2 + lam)
        cmu = min(1 - c1, (2 + 2 / (n + 1)) / ((n + 1.3) ** 2 + lam))
        cc = (4 + n / lam) / ((n + 4) + 2 / (n + 1))
        cs = (cc + 1 + 2 * min(1, (mu - 1) / lam)) / ((n + cc) + 3)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / lam - 1) - 1) + cs

        # Weights
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1 / (w @ w)

        # State
        mean = x0.copy()
        C = np.eye(n)
        Cinv = np.eye(n)
        Cpow = np.eye(n)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Precompute Cholesky of C for sampling
        def chol_update(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Symmetrize and retry
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C + np.eye(n) * 1e-10)
                return L

        L = chol_update(C)

        calls = 0

        for gen in range(budget):
            # Generate population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            for i in range(lam):
                y = mean + sigma * (L @ np.random.randn(n))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                if calls >= budget:
                    break
            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update path accumulators
            yw = w @ ys[:mu].T
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (mean - yw) / sigma
            norm_pc = np.linalg.norm(pc)
            chi_n = np.sqrt(n) * (1 - 1 / (2 * n**2) + 1 / (12 * n**4))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (mean - yw) / sigma

            # New mean
            mean = yw

            # Update covariance
            delta_hs = (norm_pc / chi_n > 1 + 4.0 / (n + 1)) and (1 - (1 - cs) * np.exp(-cs * damps)) or 0
            C = ((1 - c1 - cmu) * C +
                 c1 * (np.outer(pc, pc) + (delta_hs - 1) * C) +
                 cmu * (w @ (ys[:mu] - mean)[..., None] @ (ys[:mu] - mean).T / w.sum()))
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-10, 100)

            # Update Cholesky
            L = chol_update(C)

        # Final evaluation at best found point
        best_x = np.clip(mean, -5, 5)
        best_f = func(best_x)
        calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
