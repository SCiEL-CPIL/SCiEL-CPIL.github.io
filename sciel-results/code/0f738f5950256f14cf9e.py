"""Rewritten optimizer for BBOB f6 (Attractive sector) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with wall-clock deadline guard for the BBOB f6 task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        N = 4 + int(3 * np.log(dim + 1))
        N = max(N, 2 * dim)
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / ((dim + 1) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        x_mean = np.zeros(dim)
        sigma = 4.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            for i in range(N):
                z = rng.standard_normal(dim)
                chol_C = np.linalg.cholesky(C + 1e-10 * np.eye(dim))  # Regularize for numerical stability
                y = x_mean + sigma * chol_C @ z
                y = np.clip(y, -5, 5)
                pop.append(y)

            # Evaluate population
            fvals = np.array([func(y) for y in pop])
            calls += N

            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = np.array([pop[i] for i in idx])
            f_sorted = fvals[idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Selection: use top mu
            y_selected = y_sorted[:mu]

            # New mean
            x_new = np.sum(w[:, np.newaxis] * y_selected, axis=0)

            # Compute step in standardized space
            step = (x_new - x_mean) / sigma

            # Update evolution paths
            # hsig condition
            norm_pc = np.linalg.norm(pc)
            hsig = 1.0 if norm_pc / chiN < (1.4 + 1 / c_c) else 0.0

            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * step

            # pc update
            pc = (1 - c_c) * pc + hsig * c_c * step

            # Covariance update
            outer_sum = np.sum(w[:, np.newaxis] * np.outer(y_selected - x_mean, y_selected - x_mean), axis=0)
            C = (1 + c_1 * (hsig * (1 - c_c) / c_c + c_mu - c_1)) * C
            C -= c_1 * np.outer(pc, pc)
            C += c_mu / sigma ** 2 * outer_sum

            # Symmetrize
            C = (C + C.T) / 2

            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
