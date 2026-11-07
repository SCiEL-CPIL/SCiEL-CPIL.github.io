"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB f19 optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0

        eval_count = 0

        def safe_call(x):
            nonlocal eval_count
            if time.time() > deadline:
                return None
            val = float(func(x))
            eval_count += 1
            return val

        # CMA-ES setup
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / (w ** 2).sum()
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        eigvals = np.ones(dim)

        f_best = safe_call(x_mean)
        if f_best is None:
            return x_mean, np.inf

        best_x = x_mean.copy()

        while eval_count < budget:
            if time.time() > deadline:
                break

            # Generate population
            zs = np.random.standard_normal((lam, dim))
            pops = x_mean[None, :] + sigma * zs
            pops = np.clip(pops, -5, 5)

            # Evaluate
            fits = np.full(lam, np.inf)
            for i in range(lam):
                f_i = safe_call(pops[i])
                if f_i is None:
                    break
                fits[i] = f_i

            # Sort
            idx = np.argsort(fits)
            fits_sorted = fits[idx]
            pops_sorted = pops[idx]

            # Update mean
            x_new = np.sum(w[:mu, None] * pops_sorted[:mu], axis=0)

            # Update paths
            d = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * d
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * d

            # Covariance update
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n < 1.4 + 2 / (dim + 1)
            if not h_sigma:
                p_c = (1 - c_c) * p_c
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(pops_sorted[i] - x_mean, pops_sorted[i] - x_mean) / sigma ** 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            x_mean = x_new

            if fits_sorted[0] < f_best:
                f_best = fits_sorted[0]
                best_x = pops_sorted[0].copy()

        return best_x, f_best
# EVOLVE-BLOCK-END
