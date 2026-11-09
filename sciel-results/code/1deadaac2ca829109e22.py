"""Optimizer for BBOB-style f23 (Katsuura) using deterministic CMA-ES with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic CMA-ES with Latin Hypercube initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Track calls with a wrapper that counts and respects budget
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counting_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            fx = func(x_clipped)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x_clipped.copy()
            return fx

        # Latin Hypercube sampling for diverse initial points
        def lhs_sample(n_samples, dim, rng):
            """Generate n_samples points via Latin Hypercube Sampling in [-5,5]^dim."""
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = rng.permutation(n_samples)
                points[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples * 10.0 - 5.0
            return points

        # Phase 1: Latin Hypercube sampling (use ~15% of budget)
        n_lhs = max(10, min(40, budget // 10))
        lhs_points = lhs_sample(n_lhs, dim, rng)
        for x0 in lhs_points:
            if call_count[0] >= budget or time.time() > deadline:
                break
            counting_func(x0)

        if best_x[0] is None or call_count[0] >= budget or time.time() > deadline:
            return best_x[0], best_f[0]

        # Phase 2: Deterministic CMA-ES from the best LHS point
        # CMA-ES parameters
        mu = 4  # number of parents
        lambda_ = 2 * mu  # population size (number of offspring)
        c_c = (4 + mu / (dim + 4)) / ((dim + 1.3) + 4 + mu / (dim + 4))
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu) / ((dim + 2) ** 2 + mu_eff))
        mu_eff = mu / 2
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize CMA-ES state
        x_mean = best_x[0].copy()
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Track progress for potential restart
        best_f_cma = best_f[0]
        no_improve_count = 0
        stagnation_threshold = 50

        remaining = budget - call_count[0]
        gen = 0

        while call_count[0] < budget and time.time() < deadline:
            if remaining <= 0:
                break

            # Generate lambda_ offspring
            offspring = []
            offspring_f = []

            for i in range(lambda_):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                # Standard normal vector
                z = rng.standard_normal(dim)
                # Transform to search distribution
                y = B @ (D * z)
                x_new = np.clip(x_mean + sigma * y, -5.0, 5.0)
                f_new = counting_func(x_new)
                if call_count[0] < budget:
                    offspring.append(x_new)
                    offspring_f.append(f_new)

            if len(offspring) == 0:
                break

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in sorted_idx]
            offspring_f = [offspring_f[i] for i in sorted_idx]

            # Update best
            if offspring_f[0] < best_f[0]:
                best_f[0] = offspring_f[0]
                best_x[0] = offspring[0].copy()
                best_f_cma = offspring_f[0]
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Check for stagnation - if no improvement for too long, restart with smaller sigma
            if no_improve_count >= stagnation_threshold:
                sigma *= 0.5
                no_improve_count = 0
                if sigma < 1e-8:
                    break

            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i]
            x_mean = x_new_mean

            # Update evolution paths
            y_mean = (x_new_mean - x_mean) / sigma if sigma > 0 else np.zeros(dim)
            p_sigma = (1 - c_c) * p_sigma + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance matrix
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) / chi_n < 1.4 + 2 / (dim + 1)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (p_c @ p_c + (1 if not h_sigma else 0) * c_c * (2 - c_c) * np.eye(dim))
            for i in range(mu):
                y_i = (offspring[i] - x_mean) / sigma if sigma > 0 else np.zeros(dim)
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Update step size (CSA - Cumulative Step-size Adaptation)
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp((c_c / d_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update B and D (eigen-decomposition of C)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            D = np.sqrt(eigenvalues)
            B = eigenvectors

            gen += 1
            remaining = budget - call_count[0]

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
