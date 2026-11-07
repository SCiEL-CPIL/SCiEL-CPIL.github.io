"""CMA-ES optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based search for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to get a good starting point
        # Sample a few points to find a reasonable starting region
        initial_points = [np.zeros(dim)]
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = 3.0 * sign
                initial_points.append(x)
        for _ in range(10):
            initial_points.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        for x in initial_points:
            if check_deadline() or calls >= budget:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: CMA-ES
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Initial step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Learning rates
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_cov_plus = (2 + 2 / (mu_eff + 1)) / (dim + 3)
        c_cov_min = 2 / (dim + dim / mu_eff + 2)
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = best_x.copy()
        fx = best_f

        while calls < budget and not check_deadline():
            # Generate population
            population = []
            for i in range(lambda_):
                if check_deadline() or calls >= budget:
                    break
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                L = np.linalg.cholesky(C)
                y = L @ z
                # New candidate
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                fy_new = func(x_new)
                calls += 1
                population.append((fy_new, x_new, y, z))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda p: p[0])

            # Update best
            if population[0][0] < fx:
                fx = population[0][0]
                x = population[0][1].copy()

            # Update mean
            x_new = sum(w[i] * population[i][1] for i in range(mu))
            x_new = np.clip(x_new, -5.0, 5.0)

            # Update evolution paths
            # p_sigma
            d_x = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_x
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 / (calls - len(initial_points))))
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 / (calls - len(initial_points)))) < (1.4 + 2 / (dim + 1)) * expected_norm else 0.0

            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_x

            # Update covariance
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(w[i] * np.outer(population[i][2], population[i][2]) for i in range(mu))
            C = (1 + c_cov_plus * h_sigma * (c_c + 2 - c_c) - c_cov_min) * C + c_cov_plus * rank_one + c_cov_min * rank_mu

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x_new

            # Check for convergence
            if sigma < 1e-10:
                break

        return x, fx
# EVOLVE-BLOCK-END
