"""Rewritten optimizer using CMA-ES for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4 + int(3 * np.log(dim)), 4)  # population size
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma

        # Mean adaptation
        c_mean = (mu_eff + 2) / (dim + mu_eff + 5)

        # Covariance adaptation
        c_cov = (2 * (mu_eff - 1 + 2.0 / mu_eff)) / (
            (dim + 2) ** 2 + mu_eff
        )
        c_c = 4 / (dim + 4.1)

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Precompute eigen decomposition for Cholesky
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (
            1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2)
        )

        # Track best
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls = 1

        # Precompute for Cholesky updates
        def cholesky_decomp(A):
            """Compute Cholesky decomposition, return lower triangular L."""
            n = A.shape[0]
            L = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1):
                    s = sum(L[i, k] * L[j, k] for k in range(j))
                    if i == j:
                        L[i, i] = np.sqrt(A[i, i] - s)
                    else:
                        L[i, j] = (A[i, j] - s) / L[j, j]
            return L

        def update_C_sqrt(C):
            """Update C_sqrt and C_inv_sqrt via eigen decomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_inv_sqrt

        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Sample population
            Y = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)
            Y = np.array(Y)

            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lam

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update best
            if F[0] < f_best:
                f_best = F[0]
                x_best = Y[0].copy()

            # Weighted mean of best mu
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)

            # Update p_sigma
            # C_old_inv_sqrt * (x_new - x_mean) / sigma
            d_mean = (x_new - x_mean) / sigma
            C_inv_sqrt_dmean = C_inv_sqrt @ d_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * C_inv_sqrt_dmean

            # h_sigma: check if p_sigma is in expected range
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (
                norm_p_sigma
                / (
                    np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / (dim + c_sigma)))
                    * chi_n
                )
            )
            h_sigma = min(1.0, max(0.0, h_sigma))

            # Update p_c
            p_c = (
                (1 - c_c) * p_c
                + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_mean
            )

            # Update C
            rank_mu = np.outer(weights[:, None] * Y[:mu] - x_mean, 
                               (weights[:, None] * Y[:mu] - x_mean).T) / sigma**2
            rank_one = np.outer(p_c, p_c)
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (
                (1 + c_cov * (mu_eff - 1) - c_cov) * C
                + c_cov * (rank_mu + delta_h * rank_one)
            )

            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update x_mean
            x_mean = x_mean + c_mean * (x_new - x_mean)

            # Update sigma
            sigma = sigma * np.exp(
                c_sigma / d_sigma * (norm_p_sigma / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-8, 1e6)

            # Update C_sqrt, C_inv_sqrt
            C_sqrt, C_inv_sqrt = update_C_sqrt(C)

            # Check stagnation: if no improvement in last 50 calls, restart
            # (simple heuristic)

        return x_best, f_best
# EVOLVE-BLOCK-END
