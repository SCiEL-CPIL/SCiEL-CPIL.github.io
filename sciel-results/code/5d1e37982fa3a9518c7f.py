"""CMA-ES optimizer for BBOB-style f12 (Bent Cigar) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and covariance matrix adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        # CMA-ES parameters
        lam = 9  # population size
        mu = lam // 2  # number of parents
        # Weights
        w = np.log(mu + 1) - np.arange(1, mu + 1)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (d + 4.3)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Initial parameters
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)  # covariance matrix
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1

        gen = 0
        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            gen += 1

            # Generate population with antisymmetric sampling
            half = lam // 2
            if half * 2 == lam:
                z_half = np.random.randn(half, d)
                Z = np.vstack([z_half, -z_half])
            else:
                z_half = np.random.randn(half, d)
                Z = np.vstack([z_half, -z_half, np.random.randn(1, d)])

            # Transform: x_i = x_mean + sigma * (Z_i @ C^{1/2})
            # Compute C^{1/2} via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            X = x_mean + sigma * (Z @ C_sqrt.T)
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    F[i:] = best_f
                    break
                F[i] = func(X[i])
                call_count += 1

            # Check if we hit deadline mid-evaluation
            if time.time() > deadline:
                valid = F < best_f
                if np.any(valid):
                    idx = np.argmin(F)
                    best_f = F[idx]
                    best_x = X[idx].copy()
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Update mean
            x_new_mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update step size (CSA)
            y = (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (d + 1)) * chi_n else 0
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + c_c * h_sigma * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            delta_hs = (1 + 2 * max(0, (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) - 1) / chi_n)) * c_c
            C = (1 - c_1 - c_mu + delta_hs) * C + c_1 * np.outer(pc, pc)
            for i in range(mu):
                y_i = (X_sorted[i] - x_mean) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)

            # Update mean
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)

            # If sigma is very small, break early
            if sigma < 1e-10:
                break

        return best_x, best_f
