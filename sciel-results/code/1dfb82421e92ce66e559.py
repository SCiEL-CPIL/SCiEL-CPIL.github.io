"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the bent cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # standard rule
        mu = lam // 2  # number of parents
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = np.zeros(dim)  # Start at origin, but we'll do a few random restarts
        # Actually, let's start with a random point to avoid bad luck
        x = rng.uniform(-5, 5, dim)
        fx = func(x)

        # Better initialization: try a few random points
        for _ in range(5):
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            if fx_try < fx:
                x, fx = x_try, fx_try

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_made = 6  # 1 initial + 5 restarts

        # Main loop
        while calls_made < budget:
            # Generate lambda offspring
            B = np.linalg.cholesky(C)
            # Generate normal random numbers
            z = rng.standard_normal((lam, dim))
            y = z @ B.T
            yk = x + sigma * y

            # Evaluate offspring
            fy = np.empty(lam)
            for i in range(lam):
                yk_i = np.clip(yk[i], -5, 5)
                fy[i] = func(yk_i)
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                # Check if any offspring is better
                idx = np.argmin(fy)
                if fy[idx] < fx:
                    x = np.clip(yk[idx], -5, 5)
                    fx = fy[idx]
                break

            # Sort offspring
            idx_sort = np.argsort(fy)
            yk_sorted = yk[idx_sort]
            fy_sorted = fy[idx_sort]

            # Update mean
            x_new = np.sum(w[:, np.newaxis] * yk_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update covariance matrix
            ydiff = yk_sorted[:mu] - x
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + np.eye(dim)) + \
                c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.transpose(ydiff, (1, 0, 2)) @ ydiff, axis=2)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_sigma) * (B.T @ (x_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made)))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - c_c) * pc + hsig * c_c * (x_new - x) / sigma

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update best if improved
            if fy_sorted[0] < fx:
                x, fx = x_new, fy_sorted[0]
            else:
                x = x_new

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)

        return x, fx
# EVOLVE-BLOCK-END
