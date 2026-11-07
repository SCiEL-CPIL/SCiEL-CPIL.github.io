"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and rotation handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility per instance
        rng = np.random.default_rng(42)

        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2  # number of parents

        # Initial point: center of the box
        x_mean = np.zeros(dim)
        sigma = 1.5

        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Compute mu_eff
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Recalculate CMA parameters with correct mu_eff
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Best solution tracking
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls_made = 1

        def mat_pow_sym(M, p):
            """Compute M^p for symmetric positive definite M using eigenvalue decomposition."""
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals_clipped = np.clip(eigvals, 1e-10, None)
            return (eigvecs * eigvals_clipped ** p) @ eigvecs.T

        # Precompute chi_n
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        for iteration in range(budget - 1):
            # Check if we're running low on budget
            if calls_made + lambda_ > budget:
                # Use remaining budget for single evaluations
                while calls_made < budget:
                    # Use current best as base, take small steps
                    step = sigma * rng.standard_normal(dim)
                    y = np.clip(x_mean + step, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < f_best:
                        x_best = y.copy()
                        f_best = fy
                break

            # Generate population
            ys = np.zeros((lambda_, dim))
            fs = np.zeros(lambda_)

            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = np.clip(x_mean + sigma * C_sqrt @ z, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_made += 1

                if fs[i] < f_best:
                    x_best = y.copy()
                    f_best = fs[i]

            # Sort by fitness
            indices = np.argsort(fs)
            ys_sorted = ys[indices]
            fs_sorted = fs[indices]

            # Update mean
            w = weights
            x_new_mean = np.sum(w[:, np.newaxis] * ys_sorted[:mu], axis=0)

            # Update covariance matrix
            y_w = (ys_sorted[:mu] - x_mean) / sigma
            delta_sigma = np.linalg.norm(p_sigma) / chi_n - 1
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w @ w

            # Update correlation matrix evolution path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Rank-one update
            v = x_new_mean - x_mean
            C = (1 + c_1 * (h_sigma * (1 / c_c - 1) - 1) - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(y_w.T, y_w) * w[np.newaxis, :], axis=0)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T

            # Update square root
            C_sqrt = mat_pow_sym(C, 0.5)
            C_sqrt_inv = mat_pow_sym(C, -0.5)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update mean
            x_mean = x_new_mean

            # Clip mean to box
            x_mean = np.clip(x_mean, -5, 5)

            # Adaptive stopping: if sigma is very small, do local search
            if sigma < 1e-4:
                # Do a few local random steps
                while calls_made < budget:
                    step = 0.1 * rng.standard_normal(dim)
                    y = np.clip(x_best + step, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < f_best:
                        x_best = y.copy()
                        f_best = fy

        return x_best, f_best
# EVOLVE-BLOCK-END
