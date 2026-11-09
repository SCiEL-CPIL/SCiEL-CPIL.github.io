"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / ((dim + 2) * (dim + mu_eff))
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        sqrtC = np.eye(dim)
        inv_sqrtC = np.eye(dim)
        dcor = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_c

        call_count = 1
        remaining = budget - 1

        while remaining > 0 and time.time() < deadline:
            # Generate offspring
            z = np.random.standard_normal((lam, dim))
            # Apply covariance matrix square root
            y = z @ inv_sqrtC.T
            xk = x + sigma * y
            # Clip to box
            xk = np.clip(xk, -5, 5)

            # Evaluate
            fvals = np.array([func(xk[i]) for i in range(lam)])

            # Track best
            if fvals.min() < best_fx:
                idx = np.argmin(fvals)
                best_fx = fvals[idx]
                best_x = xk[idx].copy()

            # Sort
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            y_sorted = y[order]
            xk_sorted = xk[order]

            # Update mean
            x_new = xk_sorted[:mu] @ weights[:, np.newaxis]
            # Clip mean to box (soft constraint)
            x_new = np.clip(x_new, -5, 5)

            # Update evolution paths
            y_mean = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (inv_sqrtC @ y_mean)
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (remaining + 1)))
            hsig = hs < (1 + 4 / dim) * chi_n

            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            C += c_1 * np.outer(pc, pc)
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * y_sorted[:mu][:, np.newaxis, :] * y_sorted[:mu][:, np.newaxis, :].swapaxes(1, 2), axis=0)

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update square root of C
            sqrtC = eigvecs @ np.sqrt(np.diag(eigvals))
            inv_sqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

            x = x_new
            fx = fvals_sorted[0]
            remaining -= lam

        return best_x, best_fx
# EVOLVE-BLOCK-END
