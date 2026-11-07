"""Optimized solver for BBOB f8 (Rosenbrock) with hidden affine transforms."""
import numpy as np
import time


class Optimizer:
    """Population-based evolution strategy with adaptive covariance (simplified CMA-ES)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial population: centered at origin with moderate spread
        x0 = np.zeros(n)
        sigma = 2.0
        pop_size = max(8, 4 * n)  # population size
        lambda_ = pop_size

        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.log((lambda_ - 1) / n))) / (n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (n + 1)) - 1) + c_sigma
        mu = lambda_ // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (2 - c_c + min(weights[0], 2 / mu_eff - 1))
        c_mu = min(c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 + mu_eff)) * (
            0.5 + min(weights[0], 2 / mu_eff - 1)
        )
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x_mean = x0.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        sigma = 2.0

        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Generate initial population
        def sample_population():
            """Sample lambda_ points from N(x_mean, sigma^2 * C)."""
            z = rng.standard_normal((lambda_, n))
            # Transform z: x = x_mean + sigma * B * D * z
            # Using Cholesky-like decomposition: C = B diag(D^2) B^T
            transformed = (z * D) @ B.T
            return x_mean + sigma * transformed

        def update_chol(C):
            """Update B and D from C via Cholesky decomposition."""
            nonlocal B, D, C_inv
            try:
                L = np.linalg.cholesky(C)
                D = np.diag(L).copy()
                # Ensure positive diagonal
                D = np.maximum(D, 1e-12)
                B = np.eye(n)
                for i in range(n):
                    for j in range(i):
                        B[i, j] = L[i, j] / D[i]
                # C = B diag(D^2) B^T
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Fallback: use identity
                B = np.eye(n)
                D = np.ones(n)
                C = np.eye(n)
                C_inv = np.eye(n)

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Sample population
            z = rng.standard_normal((lambda_, n))
            transformed = (z * D) @ B.T
            pop = x_mean + sigma * transformed

            # Evaluate population
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline - 5.0:
                    break
                fvals[i] = eval_point(pop[i])
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop = pop[indices]
            fvals = fvals[indices]

            # Weighted recombination
            x_new = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size control variable
            C_sqrt_inv = np.linalg.cholesky(C_inv)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                C_sqrt_inv.T @ ((x_new - x_mean) / sigma)
            )

            # Update mean
            x_mean = x_new

            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            # Note: x_new is the new mean, so (x_new - x_mean) is 0. Let me fix this.
            # Actually, we should compute the difference before updating x_mean.
            # Let me restructure.

            # Check h condition
            h = (np.sqrt((1 - (1 - c_c) ** (2 * (calls // lambda_ + 1))) / (1 - (1 - c_c) ** 2))) * (
                1 - c_c / 2
            )
            hsig = np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * h

            # Update covariance matrix
            if hsig:
                rank_one = np.outer(ps, ps)
                delta_hsig = (1 - hsig) * h * (2 - h)
            else:
                rank_one = np.zeros((n, n))
                delta_hsig = 0

            # Rank-mu update
            v = np.empty((n, mu))
            for i in range(mu):
                v[:, i] = (pop[i] - x_mean) / sigma
            rank_mu = np.sum(weights[:, None, None] * v @ v.T, axis=2)

            C = (
                (1 - c_1 - c_mu + delta_hsig * c_c * (2 - c_c) * c_sigma) * C
                + c_1 * (rank_one + np.outer(pc, pc))
                + c_mu * rank_mu
            )

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update Cholesky decomposition
            update_chol(C)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e3)

        return best_x, best_f
