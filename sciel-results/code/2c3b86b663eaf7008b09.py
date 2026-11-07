"""Rewritten optimizer: CMA-ES with time guard for BBOB f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard and restarts for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        deadline = t0 + 280.0

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        # Initial random point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        pop_size = 4
        mu = pop_size // 2
        w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = x.copy()
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        while calls_made < budget:
            if time.perf_counter() > deadline:
                break

            # Generate population
            Y = np.zeros((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                Y[i] = x_mean + sigma * (B @ (D * z))

            # Evaluate
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if calls_made >= budget:
                    F[i] = np.inf
                    continue
                y = np.clip(Y[i], -5, 5)
                F[i] = func(y)
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()

            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * Y_sorted[i]
            x_new = np.clip(x_new, -5, 5)

            # Update step size
            y_old = (x_new - x_mean) / (sigma + 1e-30)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D @ y_old))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 50)

            # Update C
            y_c = (x_new - x_mean) / (sigma + 1e-30)
            p_c = (1 - c_c) * p_c + c_c * y_c
            delta_h = (1 - np.exp(-c_c * mu_eff / dim)) * (1 + np.linalg.norm(p_c) ** 2 / (2 * dim) - 1)

            Z = (Y_sorted[:mu] - x_mean[None, :]) / (sigma + 1e-30)
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(Z[i], Z[i])

            C = (1 - c_1 - c_mu + c_1 * delta_h) * C + c_1 * (np.outer(y_c, y_c) + C) + c_mu * H

            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B = eigvecs
            D = np.sqrt(eigvals)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
