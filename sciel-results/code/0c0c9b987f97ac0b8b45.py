"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # standard CMA-ES population size
        mu = lam // 2  # number of recombination weights
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Evolution paths
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        x = np.zeros(dim)  # mean
        sigma = 1.0  # step size
        C = np.eye(dim)  # covariance matrix
        C_sqrt = np.eye(dim)  # square root of C
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)  # evolution path for C
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1

        def time_ok():
            return time.time() < deadline

        def remaining():
            return budget - calls_made

        # Precompute C^{-1/2} using Cholesky
        def cholesky_inv_sqrt(C_mat):
            """Compute C^{-1/2} via Cholesky decomposition."""
            L = np.linalg.cholesky(C_mat + 1e-15 * np.eye(dim))
            # C^{-1/2} = L^{-T} L^{-1}
            L_inv = np.linalg.inv(L)
            return L_inv.T @ L_inv

        def update_C_sqrt(C_mat):
            """Update C^{1/2} via eigenvalue decomposition."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-15, None)
            return eigvecs @ np.diag(np.sqrt(eigvals))

        # Main CMA-ES loop
        while remaining() > lam + 5 and time_ok():
            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                Y[i] = x + sigma * (C_sqrt @ z)
                Y[i] = np.clip(Y[i], -5, 5)

            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * Y_sorted[i]

            # Update mean
            x_new = x + y_w

            # Update evolution paths
            C_inv_sqrt = cholesky_inv_sqrt(C)
            z_w = C_inv_sqrt @ y_w

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
            norm_p_sigma = np.linalg.norm(p_sigma)

            h_sig = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (calls_made / dim))) < (1.4 + 2 / (dim + 1)) * chi_n

            h = 1.0 if h_sig else 0.0
            p_c = (1 - c_c) * p_c + h * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_w

            # Update covariance
            delta_h = (1 - h) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(Y_sorted[i] - x, Y_sorted[i] - x)

            # Symmetrize
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e5)

            # Update mean
            x = x_new
            x = np.clip(x, -5, 5)

            # Update C_sqrt
            C_sqrt = update_C_sqrt(C)

            # Termination check
            if sigma < 1e-10:
                break

            # Check if we're making progress
            if calls_made % 50 == 0:
                # Check convergence
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.max() / max(eigvals.min(), 1e-15) > 1e14:
                    break

        # Use remaining budget for fine-tuning with (1+1)-ES
        if remaining() > 5 and time_ok():
            sigma_es = sigma * 0.1
            success_count = 0
            window = 5
            window_count = 0
            fx = best_f

            while remaining() > 0 and time_ok():
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                else:
                    success_count -= 1

                window_count += 1
                if window_count >= window:
                    if success_count > 0:
                        sigma_es *= 1.5
                    else:
                        sigma_es *= 0.5
                    success_count = 0
                    window_count = 0

                if sigma_es < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
