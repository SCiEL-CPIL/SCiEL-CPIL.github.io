"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization on BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim)) -> for dim=5, lambda=10
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # number of parents
        # Weights: logarithmic weights normalized to sum to 1
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size initialization
        sigma = 2.0

        # Covariance matrix: start with identity
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-3, 3, dim)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Cholesky factor of C
        chol = np.linalg.cholesky(C)

        calls = 0
        best_x = m.copy()
        best_f = func(best_x)
        calls += 1

        # Track best for return
        eval_count = 0

        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = chol @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                fy = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()

                if time.time() > deadline - 5:
                    break

            if calls >= budget or time.time() > deadline - 5:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order[:mu]]
            xs_sorted = [xs[i] for i in order[:mu]]
            fs_sorted = [fs[i] for i in order[:mu]]

            # Update mean
            m_new = m + sigma * np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)

            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (np.mean(ys_sorted, axis=0) / sigma))
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (24 * dim ** 2))
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            y_w = np.array(ys_sorted)  # (mu, dim)
            diff = (m_new - m) / sigma
            p_c = (1 - c_c) * p_c + c_c * diff

            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += w[i] * np.outer(ys_sorted[i], ys_sorted[i])

            h_sigma = (1 - np.exp(-c_sigma * (1 - np.sum(1 - c_sigma)))) / (1 - np.exp(-c_sigma))
            # Simplified: h_sigma ≈ 1 if norm_p_sigma / chi_n is close to 1
            h_sigma = 1.0 if abs(norm_p_sigma / chi_n - 1) < 1.4 * (1 + c_sigma) else 0.0

            C = (1 + c_1 + c_mu - c_c * (1 - c_c) - c_sigma * (2 - c_sigma)) * C \
                + c_1 * (rank_1 - (1 - h_sigma) * np.eye(dim)) \
                + c_mu * (rank_mu - np.eye(dim))

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2  # symmetrize

            # Recompute Cholesky
            try:
                chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-6 * np.eye(dim)
                chol = np.linalg.cholesky(C)

            # Update C_sqrt_inv (inverse of Cholesky)
            C_sqrt_inv = np.linalg.inv(chol)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
