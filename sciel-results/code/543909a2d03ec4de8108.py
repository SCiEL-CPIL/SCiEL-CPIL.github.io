"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population for rotated, anisotropic Weierstrass minima."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        # Initial mean: uniform in [-4, 4] (minimizer lies in [-4,4]^5)
        rng = np.random.default_rng(42)
        mean = rng.uniform(-4.0, 4.0, n)

        # Initial sigma: scale to explore the box
        sigma = 2.0

        # Covariance matrix: identity
        C = np.eye(n)
        Cinv = np.eye(n)
        C_sqrt = np.eye(n)
        Cinv_sqrt = np.eye(n)

        # Evolution path
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Step size control parameters
        cmu = 1.0 / (4 + n / 2)
        cc = (4 + n / 3) / (n + 4 + 2 * n / (n + 4))
        cs = (2 + cmu * (n / 2 - 1)) / (n + 2 + cmu * (n / 2 - 1))
        c1 = (2 - cmu + 2 / (n + 1.3) ** 2) / ((n + 1.3) ** 2 + 2)
        cmu_ = cmu / 2
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cmu
        dampc = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 2) + cc
        mu_eff = 1 / np.sum(1.0 / np.arange(1, mu + 1))
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Track best solution
        best_x = mean.copy()
        best_f = func(best_x)

        calls = 1

        # Precompute chi_n
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample population
            X = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                X[i] = mean + sigma * (C_sqrt @ z)
                X[i] = np.clip(X[i], -5.0, 5.0)

            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Weighted mean of top mu
            new_mean = np.zeros(n)
            for i in range(min(mu, lam)):
                new_mean += weights[i] * X_sorted[i]

            # Update evolution paths
            # Inverse square root of C for the step
            dmean = new_mean - mean
            step = dmean / sigma if sigma > 0 else np.zeros(n)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv_sqrt @ step)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget + 1)))) < (1.4 + 2 / (n + 1)) * chi_n

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * step

            # Update covariance
            delta_d = (1 + cc * (2 - hsig)) * dmean / sigma - 1
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(min(mu, lam)):
                zi = (X_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c1 * (2 - 1) - c1 - cmu_) * C + c1 * rank_one + cmu_ * rank_mu
            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5.0, 5.0)

            # Update C_sqrt and Cinv_sqrt periodically (every few iterations)
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            Cinv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            Cinv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
