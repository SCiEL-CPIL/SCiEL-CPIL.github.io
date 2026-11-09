"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point in center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)  # Inverse covariance matrix
        p_sigma = np.zeros(dim)  # Evolution path for sigma
        p_c = np.zeros(dim)  # Evolution path for C

        # Learning rates
        mu = None  # Will be set based on population size
        weights = None
        mu_eff = None
        c_sigma = None
        d_sigma = None
        c_c = None
        c_1 = None
        c_mu = None
        chi_n = None

        def compute_params(lam):
            nonlocal mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            weights = w
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def sample_and_evaluate(sigma_current, C_current, x_current, lam):
            """Sample lam points and evaluate them."""
            points = []
            values = []
            # Generate random directions
            B = np.linalg.cholesky(C_current)
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_current + sigma_current * (B @ z)
                y = np.clip(y, -5, 5)
                points.append(y)
                values.append(func(y))
            return np.array(points), np.array(values)

        # Adaptive population schedule
        # Start large, shrink as we go
        used = 1  # Already used 1 evaluation
        iteration = 0

        while used < budget:
            iteration += 1
            # Determine population size based on remaining budget and progress
            remaining = budget - used
            if remaining > 100:
                lam = min(25, remaining)
            elif remaining > 50:
                lam = min(15, remaining)
            elif remaining > 20:
                lam = min(10, remaining)
            elif remaining > 10:
                lam = min(5, remaining)
            else:
                lam = remaining

            if lam < 2:
                # Just do a few single-step evaluations
                for _ in range(lam):
                    if used >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    used += 1
                    if fy < best_fx:
                        best_x = y.copy()
                        best_fx = fy
                break

            # Compute parameters for this population size
            compute_params(lam)

            # Sample and evaluate
            points, values = sample_and_evaluate(sigma, C, x, lam)
            used += lam

            # Sort by fitness
            order = np.argsort(values)
            points = points[order]
            values = values[order]

            # Update best
            if values[0] < best_fx:
                best_x = points[0].copy()
                best_fx = values[0]

            # Weighted mean
            x_new = np.sum(weights[:, None] * points[:mu], axis=0)

            # Update evolution paths
            y_sigma = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (used - 1))) <
                            (1.4 + 2 / (dim + 1)) * chi_n) else 0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

            # Update covariance matrix
            outer = np.outer(y_sigma, y_sigma)
            rank1 = c_1 * (outer + (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c))
            rankmu = c_mu * np.sum(weights[:mu, None, None] * np.outer(points[:mu, None] - x, points[:mu, None] - x) / (sigma ** 2), axis=0)
            C = (1 + c_1 + c_mu - c_1) * C + rank1 + rankmu
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
