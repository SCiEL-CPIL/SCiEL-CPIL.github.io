"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Cap lambda so we don't exceed budget
        max_lam = min(lam, max(2, budget // 10))
        lam = max(2, max_lam)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fvals = np.array([func(x)])
        best_x = x.copy()
        best_f = fvals[0]

        # CMA-ES parameters
        mu = lam // 2  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for efficient updates
        # Use Cholesky or eigendecomposition; we'll use eigendecomposition
        # Actually, let's use the standard CMA-ES update with Cholesky
        # For simplicity and stability, we'll use the eigendecomposition approach
        # But that's expensive. Let's use the standard approach with Cholesky.

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute
        # We need sqrt(C) for sampling. We'll use Cholesky decomposition.
        # But Cholesky can fail if C is not positive definite. We'll regularize.

        # Count function evaluations
        calls = 1

        # Main loop
        while calls < budget:
            # Generate population
            # Get Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Sample
            z = rng.standard_normal((lam, dim))
            y = x[None, :] + sigma * (z @ L.T)
            # Clip to bounds
            y = np.clip(y, -5, 5)

            # Evaluate
            fvals = np.array([func(yi) for yi in y])
            calls += lam
            if calls >= budget:
                # Check if any improved
                for i in range(len(fvals)):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = y[i].copy()
                break

            # Sort
            idx = np.argsort(fvals)
            y = y[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = y[0].copy()

            # Weighted mean
            x_new = np.sum(weights[:, None] * y[:mu], axis=0)

            # Update paths
            # h_sigma
            sqrt_mu_eff = np.sqrt(mu_eff)
            # ||L^{-1}||
            # Approximate norm
            try:
                L_inv = np.linalg.inv(L)
                norm_L_inv = np.linalg.norm(L_inv, 2)
            except np.linalg.LinAlgError:
                norm_L_inv = 1.0

            h_sigma = (1 - np.exp(-c_sigma * (1 - c_c / 2) * dim / (dim + 1))) if norm_L_inv > 1e-10 else 1.0
            # Actually, the standard formula is:
            # h_sigma = (1 - exp(-c_sigma * (1 - c_c/2))) if ||L^{-1}|| < ... 
            # Let's use a simpler check
            if norm_L_inv < 1e-10:
                h_sigma = 1.0
            else:
                h_sigma = (1 - np.exp(-c_sigma * (1 - c_c / 2))) if (norm_L_inv / dim) < (1 + 2 / (dim + 1)) * (1 - np.exp(-c_sigma * (1 - c_c / 2))) else 0.0

            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(mu_eff) * np.linalg.solve(L.T, (x_new - x) / sigma)

            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * np.linalg.solve(L, (x_new - x) / sigma)

            # Covariance matrix update
            delta_h = (1 / h_sigma - 1) * np.outer(p_sigma, p_sigma)
            C = (1 + c_1 * c_mu * (1 - c_c / 2) - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C - delta_h) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y[:mu] - x, y[:mu] - x) / (sigma ** 2), axis=0)

            # Symmetrize
            C = (C + C.T) / 2

            # Update x and sigma
            x = x_new
            sigma *= np.exp(c_sigma / dim * (np.linalg.norm(p_sigma) / sqrt_mu_eff - 1))

            # Ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-10, 1e10)

        return best_x, best_f
# EVOLVE-BLOCK-END
