"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # standard CMA-ES population size

        # Initial point and function value
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Step size (sigma) and covariance matrix
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        D = np.ones(n)  # diagonal of sqrt(C)

        # Strategy parameters
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Mean evolution path
        ps = np.zeros(n)
        hsig = (1 + 2 / (n + 1)) ** cs  # threshold for hsig

        # Initial evaluation count
        calls = 1
        iter_count = 0

        while calls < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((pop_size, n))
            fvals = np.zeros(pop_size)

            for i in range(pop_size):
                # Transform using Cholesky-like factorization (using D and C_inv approx)
                # For simplicity, use D * C_inv^0.5 * z, but we'll use a simpler approach
                z = rng.standard_normal(n)
                # Approximate: y = x + sigma * D @ (C_inv ** 0.5) @ z
                # For now, use diagonal approximation for speed
                y = x + sigma * D * z
                y = np.clip(y, -5, 5)
                fvals[i] = func(y)
                calls += 1
                ys[i] = y

                # Check deadline mid-loop
                if calls >= self.budget or time.time() > deadline:
                    break

            # If we broke early, adjust pop_size for this iteration
            actual_pop = min(i + 1, pop_size)
            if actual_pop < 2:
                break

            ys = ys[:actual_pop]
            fvals = fvals[:actual_pop]

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = ys_sorted[0].copy()

            # Update mean
            x_new = np.zeros(n)
            for i in range(min(mu, actual_pop)):
                x_new += weights[i] * ys_sorted[i]
            x = x_new

            # Update evolution path ps
            # Need C_inv^0.5, approximate with 1/D
            C_inv_sqrt = (1.0 / D)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (
                (x - x_new) / sigma
            )

            # Update sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iter_count + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (
                (x - x_new) / sigma
            )
            sigma *= np.exp(cs / n * (np.linalg.norm(ps) / np.sqrt(2 * n * (1 - (1 - cs) ** (iter_count + 1))) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix C
            delta_hsig = (1.0 if hsig else 0.0)
            C = (1 + c1 * delta_hsig * (2 - c1) + c1 + cmu) * C
            C -= (c1 * delta_hsig * (2 - c1) + c1 + cmu) * np.outer(ps, ps)

            for i in range(min(mu, actual_pop)):
                yi = C_inv_sqrt @ ((ys_sorted[i] - x_new) / sigma)
                C += cmu * weights[i] * np.outer(yi, yi)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update D and C_inv
            D = np.sqrt(np.diag(C))
            C_inv = np.linalg.inv(C)

            iter_count += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
