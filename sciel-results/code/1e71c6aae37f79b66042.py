"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware budget management for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES
        # Start from a reasonable point; Rosenbrock's minimum is at (1,...,1)
        # but the function is shifted/rotated, so start from origin or random
        x_mean = np.zeros(n)
        sigma = 2.0

        # Population size
        lam = 4 + int(3 * np.log(n))  # standard CMA-ES population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_mu = sum(weights[:min(mu, int(0.5 * mu))]) / (1 + sum(weights[:min(mu, int(0.5 * mu))]))
        c_sigma = (c_mu + 2) / (n + c_mu)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = 2 * (c_mu - 2 + c_1) / ((n + 2) ** 2 + c_mu)

        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial evaluation
        f0 = evaluate(x_mean)

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            y = rng.standard_normal((lam, n))
            z = y @ B.T
            x_pop = np.array([x_mean + sigma * z[i] for i in range(lam)])

            # Evaluate population
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lam)])

            # Sort by fitness
            idx = np.argsort(f_pop)
            x_pop = x_pop[idx]
            f_pop = f_pop[idx]

            # Update mean
            x_new = sum(weights[i] * x_pop[i] for i in range(mu))

            # Update evolution paths
            C_sqrt_inv = np.linalg.cholesky(np.linalg.inv(C))
            pc = (1 - c_c) * pc + c_c * C_sqrt_inv.T @ (x_new - x_mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * c_mu) * C_sqrt_inv @ (x_new - x_mean) / sigma

            # Damping factor for rank-mu update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - budget // 10)))) <= \
                   (1 + 2 / (n + 1)) * chi_n

            # Update covariance
            delta_sig = (hsig - 1) * c_sigma * d_sigma
            C = (1 + c_1 * delta_sig * c_sigma + c_1 * c_sigma) * C - c_1 * np.outer(pc, pc) + \
                c_mu * sum(weights[i] * np.outer(x_pop[i] - x_mean, x_pop[i] - x_mean) for i in range(mu))

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Project to positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
