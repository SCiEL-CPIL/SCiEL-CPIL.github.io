"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with proper implementation and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu = lam // 2
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) * (mu_eff + 5)) - 1))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1

        # Precompute square root of C
        def sqrtm(C_mat):
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        B = sqrtm(C)

        for gen in range(total_budget - 1):
            if not time_ok():
                break

            # Generate population
            Z = rng.standard_normal((lam, dim))
            Y = x_mean + sigma * (Z @ B.T)
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_used += lam

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Compute weighted mean of top mu
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * Y_sorted[i]

            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w - x_mean) / sigma

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update evolution path for C
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_mean) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_c, p_c) + delta_h * C
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                diff = Y_sorted[i] - x_mean
                rankmu += weights[i] * np.outer(diff, diff)

            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            C = (C + C.T) / 2

            # Update B
            B = sqrtm(C)

            # Update mean
            x_mean = y_w

        return best_x, best_f
# EVOLVE-BLOCK-END
