"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for bent cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box (0,0,0,0,0) is a reasonable start
        x = np.zeros(dim)
        fx = func(x)

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix and its Cholesky-like decomposition
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Step size
        sigma = 2.0  # initial step size

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial best
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        for _ in range(budget - calls_made):
            if calls_made >= budget:
                break

            # Generate population
            ys = []
            ys_func = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                calls_made += 1
                fy = func(y)
                ys_func.append(fy)

            # Sort by fitness
            indices = np.argsort(ys_func)
            ys_sorted = [ys[i] for i in indices]
            ys_func_sorted = [ys_func[i] for i in indices]

            # Update best
            if ys_func_sorted[0] < best_f:
                best_f = ys_func_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean of selected points
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * ys_sorted[i]

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (m - x) / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)

            h_sig = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

            # Update mean
            x = m

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            C = ((1 + c_1 + c_mu - c_1) * C)  # base decay
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (ys_sorted[i] - m) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                # Fallback: reset to identity if decomposition fails
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e3)

            # Keep x within bounds
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
