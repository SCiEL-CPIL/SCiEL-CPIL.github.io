"""Rewritten optimizer: CMA-ES with time guard and adaptive sigma."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin within 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: use a few random starts to pick the best
        rng = np.random.default_rng(42)
        n_starts = 5
        starts = rng.uniform(-4, 4, (n_starts, dim))
        start_vals = np.array([func(s) for s in starts])
        x = starts[np.argmin(start_vals)].copy()
        best_val = start_vals.min()
        calls_made = n_starts

        # Initial sigma
        sigma = 2.0

        # Covariance matrix and related
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        c_sqrt_inv = 1.0 / np.sqrt(1 - (1 - c_c) ** 2)

        def mat_sqrt_inv(C):
            """Compute inverse square root of covariance matrix via eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        def mat_sqrt(C):
            """Compute square root of covariance matrix via eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        C_sqrt_inv = mat_sqrt_inv(C)
        C_sqrt = mat_sqrt(C)

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda_ samples
            z = rng.standard_normal((lambda_, dim))
            y = x[None, :] + sigma * (z @ C_sqrt)
            # Clip to box
            y = np.clip(y, -5, 5)

            # Evaluate
            vals = np.array([func(yp) for yp in y])
            calls_made += lambda_

            # Track best
            idx_best = np.argmin(vals)
            if vals[idx_best] < best_val:
                best_val = vals[idx_best]
                x_best = y[idx_best].copy()

            # Sort by fitness
            order = np.argsort(vals)
            y_sel = y[order[:mu]]
            vals_sel = vals[order[:mu]]

            # Update mean
            x_new = np.sum(weights[:, None] * y_sel, axis=0)

            # Update step-size control (CSA)
            y_sigma = (C_sqrt_inv @ (x_new - x)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < (1.4 + 2 / (dim + 1)) * (1 - c_sigma)

            # Adapt sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            delta_h = (1 - hsig) * c_c * (2 - c_c)

            rank_1 = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * np.outer(y_sel - x, y_sel - x), axis=0)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            C_sqrt_inv = mat_sqrt_inv(C)
            C_sqrt = mat_sqrt(C)

            x = x_new

        return x, best_val
# EVOLVE-BLOCK-END
