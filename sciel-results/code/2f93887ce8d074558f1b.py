"""Rewritten optimizer for BBOB f11 (Discus) with per-dimension CMA-ES."""
import numpy as np
import time


class Optimizer:
    """Per-dimension CMA-ES with success rule for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at origin (center of search space)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Per-dimension step sizes
        sigma = np.full(dim, 2.0)

        # CMA-ES style parameters
        c_sigma = 0.5 / (dim + 3)
        c_c = (4 + dim / dim) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((dim + 1.5) / (dim + 3)) ** 2)

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weight vector
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial covariance diagonal (per-dimension variances)
        C_diag = np.ones(dim)

        # Budget tracking
        calls_used = 1

        for gen in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Apply per-dimension scaling and covariance diagonal
                step = sigma * np.sqrt(C_diag) * z
                offspring[i] = np.clip(x + step, -5, 5)

            # Evaluate offspring
            f_values = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with large value
                    f_values[i:] = np.inf
                    break
                f_values[i] = func(offspring[i])
                calls_used += 1

            # Update best
            min_idx = np.argmin(f_values)
            if f_values[min_idx] < best_f:
                best_f = f_values[min_idx]
                best_x = offspring[min_idx].copy()

            # Sort by fitness
            order = np.argsort(f_values)
            y_sorted = offspring[order]
            f_sorted = f_values[order]

            # Mean of best mu individuals
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            sigma_old = sigma.copy()
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (y_w - x) / np.maximum(sigma_old, 1e-10)
            )

            # Update step sizes (per-dimension success rule)
            for d in range(dim):
                if p_sigma[d] ** 2 / dim < 1 + 2.0 / (dim + 2):
                    sigma[d] *= np.exp(0.5 * c_sigma * (
                        np.linalg.norm(p_sigma) / np.sqrt(dim) - 1.0
                    ))
                # Clip sigma to reasonable range
                sigma[d] = np.clip(sigma[d], 1e-8, 10.0)

            # Update covariance diagonal (simplified)
            for d in range(dim):
                var = np.sum(weights * (y_sorted[:mu, d] - y_w[d]) ** 2)
                C_diag[d] = (1 - c_1 - c_mu) * C_diag[d] + c_1 * C_diag[d] + c_mu * var / max(sigma[d] ** 2, 1e-10)
                C_diag[d] = np.clip(C_diag[d], 0.1, 10.0)

            # Move center
            x = np.clip(y_w, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check deadline again
            if time.time() > deadline:
                break

        return best_x, best_f
