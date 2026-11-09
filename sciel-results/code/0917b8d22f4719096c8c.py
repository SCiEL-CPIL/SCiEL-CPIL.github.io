"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance adaptation for rotated landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda
        c_sigma = (pop_size + 2) / ((n + 1.3) ** 2 + pop_size)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / n) / (n + 4 + 2 * pop_size / n)
        c_1 = 2 / ((n + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - n - 2) / ((n + 2) ** 2 + pop_size))
        mu_w = np.arange(pop_size // 2 + 1)
        mu_w = mu_w ** -0.5
        mu_w = mu_w / mu_w.sum()
        mu_eff = 1.0 / (mu_w ** 2).sum()

        # Initial state
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        sigma = 1.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def update_chol(C, max_iter=100):
            """Compute Cholesky factorization with fallback."""
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Add small jitter
                C = C + 1e-10 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                    return L
                except np.linalg.LinAlgError:
                    # Fall back to eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    return L

        for gen in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Update step size
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                C_inv_sqrt @ (np.sort(np.array([func.__self__ if hasattr(func, '__self__') else None] * pop_size)) - x) if False else C_inv_sqrt @ (np.zeros(n))  # placeholder, will fix below
            )

            # Generate population
            ys = []
            for i in range(pop_size):
                y = x + sigma * (C_sqrt @ rng.standard_normal(n))
                y = np.clip(y, -5, 5)
                ys.append(y)

            # Evaluate population
            fs = np.array([func(y) for y in ys])
            calls += pop_size

            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = ys[idx_best].copy()

            # Sort by fitness
            order = np.argsort(fs)
            y_sorted = np.array(ys)[order]
            f_sorted = fs[order]

            # Weighted mean
            y_mean = np.zeros(n)
            for i in range(pop_size // 2 + 1):
                y_mean += mu_w[i] * y_sorted[i]

            # Update covariance
            diff = y_sorted[:pop_size // 2 + 1] - x
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c)) * (np.eye(n) if np.dot(pc, pc) / n > (1 + 4 / n) * c_c ** 2 else 0)) + c_mu * (diff.T @ np.diag(mu_w[:pop_size // 2 + 1]) @ diff)

            # Ensure positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y_mean - x))
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (y_mean - x))

            # Move mean
            x = y_mean

            # Update Cholesky
            C_sqrt = update_chol(C)
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.linalg.pinv(C_sqrt)

            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
