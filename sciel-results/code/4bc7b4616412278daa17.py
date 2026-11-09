"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 320.0  # 10s safety margin before 330s kill
        n = self.dim
        rng = np.random.default_rng(12345)

        # Population size
        lam = max(10, int(4 + 3 * np.log(n)))

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)

        # Initial step size
        sigma = 2.0

        # CMA-ES state
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)  # C^(-1/2)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Parameters
        mu = lam // 2
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Call counter
        calls = 1
        call_count = 0

        def count_call(x_val):
            nonlocal calls, call_count
            call_count += 1
            return func(x_val)

        # Precompute sqrt weights
        sqrt_w = np.sqrt(weights[:mu])

        while call_count < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update C^(-1/2) via eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

            # Sample lambda points
            z = rng.standard_normal((lam, n))
            y = z @ C_inv_sqrt
            ys = np.array([np.clip(x + sigma * y[i], -5, 5) for i in range(lam)])

            # Evaluate
            fvals = np.array([count_call(ys[i]) for i in range(lam)])
            call_count += lam

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = y[order]
            fvals_sorted = fvals[order]

            # Weighted mean of best mu
            y_w = np.sum(weights[:mu, None] * y_sorted[:mu], axis=0)

            # Update center
            x = x + c_c * sigma * y_w

            # Update sigma path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w) / sigma

            # Update covariance (rank-mu update)
            delta_sigma = (np.linalg.norm(p_sigma) / chi_n - 1) * d_sigma
            C = ((1 - c_c * (2 - c_c) + c_sigma * delta_sigma) * C
                 + c_c * (2 - c_c) * np.outer(p_c, p_c)
                 + c_sigma * delta_sigma * C)

            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update best
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x_best = ys[order[0]].copy()

        return x, fx
# EVOLVE-BLOCK-END
