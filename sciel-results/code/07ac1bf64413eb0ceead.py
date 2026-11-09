"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rastrigin-like multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin (300s limit)
        budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # We'll do a series of CMA-ES runs with restarts
        # Strategy: Start with a moderately sized population, restart when
        # progress stalls or after a fixed number of iterations

        rng = np.random.default_rng(42)

        # Initial exploration: evaluate a few random points to get a sense of landscape
        # Use a small grid/random sample
        n_init = 20
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            init_points.append((fx, x))
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            if time.time() > deadline:
                break

        # Sort by fitness, use best as starting point
        init_points.sort(key=lambda t: t[0])
        best_start = init_points[0][1].copy()
        best_start_f = init_points[0][0]

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        # Initial sigma: start relatively large to explore
        sigma = 2.0

        # CMA-ES state
        x_mean = best_start.copy()
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # We'll use simple update for now
        step_size = 1.0

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        mu = lam // 2  # number of selected
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Track progress for restart decision
        best_in_run = best_start_f
        iterations_since_improvement = 0
        max_iter_no_improve = 100
        total_calls = n_init

        # Main loop with restarts
        restart_count = 0
        while total_calls < budget:
            if time.time() > deadline:
                break

            # Check for restart condition
            if restart_count > 0 and (iterations_since_improvement > max_iter_no_improve):
                # Restart: reset CMA-ES state around global best
                x_mean = global_best_x.copy()
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(0.1, sigma * 0.5)  # Decrease sigma on restart
                iterations_since_improvement = 0
                best_in_run = global_best_f
                restart_count += 1

            # Generate population
            offspring = []
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                # Use simple Cholesky-like approach
                z = rng.standard_normal(dim)
                # Apply C via matrix square root (approximate with C itself for simplicity)
                # Better: use eigendecomposition periodically
                y = x_mean + sigma * (C @ z)  # This is not quite right, let's use proper method
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append((fy, y))
                if fy < global_best_f:
                    global_best_f = fy
                    global_best_x = y.copy()
                if fy < best_in_run:
                    best_in_run = fy
                    iterations_since_improvement = 0
                else:
                    iterations_since_improvement += 1

                if time.time() > deadline:
                    break

            if total_calls >= budget or time.time() > deadline:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            x_mean = new_mean

            # Update covariance matrix
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (offspring[i][1] - x_mean) / sigma

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - 1 / c_c) * C) \
                + c_mu * (v @ np.diag(weights) @ v.T)

            # Ensure C is positive definite (add regularization if needed)
            C = (C + C.T) / 2  # Symmetrize
            # Add small regularization
            C += 1e-12 * np.eye(dim)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (C @ (v @ weights)) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                  (C @ (v @ weights)) / sigma

            # Sigma adaptation (1/5 rule variant)
            # Simple approach: adjust sigma based on progress
            if best_in_run < global_best_f + 1e-10:
                # Good progress
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)

            # If we've been stuck for a while, trigger restart
            if iterations_since_improvement > max_iter_no_improve:
                continue  # Will trigger restart at top of loop

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
