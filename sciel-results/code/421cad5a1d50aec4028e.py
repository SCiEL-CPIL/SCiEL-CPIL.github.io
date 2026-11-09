"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with covariance adaptation and 1/5 success rule, time-aware."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 2.0

        calls_made = 1
        best_x = x.copy()
        best_f = fx

        # Covariance matrix adaptation
        C = np.eye(dim)
        # Memory of successful directions
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)      # evolution path for covariance
        c_sigma = 1.0 / (dim + 6)
        c_c = (2 + 2.0 / (dim + 1)) / (dim + 4)
        mu_sigma = c_sigma * (1 + 2.0 / max(dim - 1, 1))
        mu_c = c_c * (1 + 2.0 / max(dim - 1, 1))

        # Precompute Cholesky for efficiency
        try:
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(dim)

        for i in range(self.budget - 1):
            # Check time budget
            if time.time() > deadline:
                break

            # Generate candidate using covariance
            z = rng.standard_normal(dim)
            # Transform z using Cholesky of C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                d = L @ z
            except np.linalg.LinAlgError:
                d = z

            y = np.clip(x + sigma * d, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: update paths and covariance
                delta = (y - x) / max(sigma, 1e-10)

                # Update sigma evolution path
                p_sigma = (1 - mu_sigma) * p_sigma + mu_sigma * delta

                # Update covariance evolution path
                p_c = (1 - mu_c) * p_c + mu_c * delta

                # Update covariance matrix
                C = (1 - c_c) * C + c_c * np.outer(delta, delta)

                # Regularize C to maintain positive definiteness
                C = 0.5 * (C + C.T)
                # Add small regularization
                C += 1e-8 * np.eye(dim)

                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.87

            # Periodically recompute Cholesky if needed
            if i % 50 == 49:
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
