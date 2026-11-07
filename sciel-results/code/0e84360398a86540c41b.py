"""CMA-ES optimizer for BBOB f12 (Bent cigar)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        rng = np.random.default_rng(0)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w * w).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_mu = (2 + (mu_eff - 2) / (dim + 2)) * c_sigma * (2 - c_sigma)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.linalg.inv(C)
        C_sqrt = np.linalg.cholesky(C)
        C_inv_sqrt = np.linalg.cholesky(C_inv)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(dim)
                X[i] = x_mean + sigma * (C_sqrt @ z)
                X[i] = np.clip(X[i], -5, 5)
                F[i] = func(X[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Use only the mu best solutions
            X_mu = X_sorted[:mu]

            # New mean
            x_old = x_mean.copy()
            x_mean_new = np.sum(w[:, None] * X_mu, axis=0)

            # Update step-size control
            y_mean = (x_mean_new - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_mean)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 15.0)

            # Update evolution paths and covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Check for h_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1) / lam)) <
                       (1.4 + 2 / (dim + 1)) * chi_n)

            # Update covariance matrix
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * np.outer(p_c, p_c)
            C += c_mu * (np.sum(w[:, None] * np.outer((X_mu - x_old) / sigma, (X_mu - x_old) / sigma), axis=0) - C)

            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                C_inv_sqrt = np.linalg.cholesky(C_inv + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 0.5)

            x_mean = x_mean_new
            gen += 1

        return best_x, best_f
