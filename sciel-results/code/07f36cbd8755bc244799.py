"""CMA-ES optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard CMA-ES with lambda = 4 + floor(3 * log10(dim))
        lam = 4 + int(3 * np.log10(dim))
        # mu = floor(lam / 2)
        mu = lam // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chi-lambda for normal distribution
        chi_lam = np.sqrt(lam) * (1 - 1 / (4 * lam) + 1 / (24 * lam ** 2))

        # Initial state
        # Start at origin, which is a reasonable center for the box
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # We'll use the Cholesky of C_inv for sampling
        # Actually, for CMA-ES we sample from N(mean, sigma^2 * C)
        # We can use the Cholesky decomposition of C for sampling
        # Or we can use the eigendecomposition
        # Let's use a simpler approach: maintain C and use its Cholesky

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Best so far
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1

        # Precompute some constants
        h_sigma = (1 + 2 / (mu_eff + dim) + 1 / mu_eff) * c_sigma
        h_one = (1 + 2 / (dim + 1) + c_1) * c_c

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample lambda points
            # For efficiency, we'll sample all lambda points at once
            # x_i = mean + sigma * C^(1/2) * z_i
            # We need C^(1/2). We can compute the Cholesky of C.
            # But C is symmetric positive definite, so we can use np.linalg.cholesky

            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                C_chol = np.eye(dim)

            # Generate random normal vectors
            Z = np.random.default_rng().standard_normal((lam, dim))
            # Transform to get X ~ N(mean, sigma^2 * C)
            X = np.array([mean + sigma * C_chol @ z for z in Z])
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate all points
            F = np.array([func(x) for x in X])
            calls += lam

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Compute new mean (weighted average of top mu)
            y = X_sorted[:mu]  # (mu, dim)
            new_mean = mean + sigma * (w @ y)  # w is (mu,), y is (mu, dim) -> (dim,)

            # Update path variables
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * (new_mean - mean) / sigma
            # We need C^(-1/2). Let's compute it from the Cholesky.
            # C = L L^T, so C^(-1/2) is more complex.
            # Alternative: use the eigendecomposition or just approximate.
            # For simplicity, let's use the inverse of C_chol @ C_chol.T
            # Actually, C^(-1/2) can be computed as (C_chol.T^(-1)) @ C_chol^(-1) ... no.
            # Let's use a different approach: maintain the inverse Cholesky.
            # Or, we can skip the exact C^(-1/2) and use a simpler update.
            # Actually, let's just compute C_inv via np.linalg.inv(C) and use its Cholesky.

            # Recompute C_inv
            try:
                C_inv = np.linalg.inv(C)
                C_inv_chol = np.linalg.cholesky(C_inv)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
                C_inv_chol = np.eye(dim)

            y_bar = (w @ y) / sigma  # (dim,)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_chol @ y_bar)

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_lam * (np.linalg.norm(p_sigma) / chi_lam - 1))

            # Update C
            # delta_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(calls+1)/mu_eff}) < (1.4 + 2/(dim+1)) * chi_lam else 0
            delta_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls / mu_eff))) < (1.4 + 2 / (dim + 1)) * chi_lam else 0.0

            # p_c = (1 - c_c) * p_c + h_c * sqrt(mu_eff) * y_bar
            h_c = 0.5 if delta_sigma else 0.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(mu_eff) * y_bar

            # delta_h = 1 if ||p_c|| / sqrt(1 - (1-c_c)^{2*(calls+1)/mu_eff}) < (1.4 + c_c) * chi_lam else 0
            delta_h = 1.0 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * calls / mu_eff))) < (1.4 + c_c) * chi_lam else 0.0

            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - delta_h) * c_c (2 - c_c) I) + c_mu * sum w_i y_i y_i^T
            y_outer = np.array([np.outer(yi, yi) for yi in y])  # (mu, dim, dim)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - delta_h) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * (w @ y_outer)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update mean
            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
