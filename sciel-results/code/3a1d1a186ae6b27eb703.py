"""Optimizer for BBOB f21 (Gallagher 101 peaks) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))  # 12 for dim=5
        # Initial sigma
        sigma = 1.0
        # Initial mean
        x_mean = np.zeros(dim)
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        # Eigendecomposition of C
        D_inv = np.eye(dim)
        B = np.eye(dim)
        # Step-size adaptation
        c_sigma = (np.sqrt(lam) - 1) / (dim + 3) + 2 * (min(2, (lam - 2) / dim) - 1) / (dim + 1)
        d_sigma = c_sigma + 2 * max(0, np.sqrt((lam + 1 - 2) / (dim + 1) - 1) - 1)
        # Mean update
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) - 1) / ((dim + 2 + mu_eff) * (lam / 2 + 1) - 2))
        # mu_eff
        mu = int(lam / 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) - 1) / ((dim + 2 + mu_eff) * (lam / 2 + 1) - 2))
        # Damping for mean update
        c_mean = c_c + c_mu
        # Step size
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        best_x = x_mean.copy()
        best_f = float('inf')
        calls = 0

        # Initial evaluation
        x_mean = rng.uniform(-4, 4, dim)
        f_mean = func(x_mean)
        calls += 1
        best_x = x_mean.copy()
        best_f = f_mean

        # Path for sigma adaptation
        p_sigma = np.zeros(dim)
        # Path for covariance adaptation
        p_c = np.zeros(dim)

        # Weights for rank-mu update
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) - 1) / ((dim + 2 + mu_eff) * (lam / 2 + 1) - 2))

        # Evaluate initial population
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(dim)
                xi = x_mean + sigma * (B @ (D_inv @ z))
                xi = np.clip(xi, -5, 5)
                xs[i] = xi
                fs[i] = func(xi)
                calls += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = xi.copy()

            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * (xs_sorted[i] - x_mean)
            x_new_mean = x_mean + c_mean * y_w

            # Update path for sigma
            y_sigma = (B @ (D_inv @ (B.T @ (x_new_mean - x_mean)))) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            norm_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_sigma / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) *
                       chi_n < (1.4 + 2 / (dim + 1)))
            sigma *= np.exp(c_sigma / d_sigma * (norm_sigma / chi_n - 1))

            # Update covariance
            y_c = (B @ (D_inv @ (B.T @ (x_new_mean - x_mean))))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Rank-1 update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C += c_1 * (np.outer(y_c, y_c) + delta_h * C)

            # Rank-mu update
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (xs_sorted[i] - x_mean) / sigma
            v = B @ (D_inv @ v)
            C += c_mu * (w @ (v @ v.T)) - c_mu * mu_eff * C

            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)

            # Update eigendecomposition periodically
            if calls % lam == 0:
                C = (C + C.T) / 2
                eigenvalues, B = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                D_inv = 1.0 / np.sqrt(eigenvalues)

        return best_x, best_f
# EVOLVE-BLOCK-END
