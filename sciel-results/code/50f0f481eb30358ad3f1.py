"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and multi-start for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 300s wall-clock

        calls = 0
        best_f = np.inf
        best_x = None

        def tracked_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= self.budget:
                raise TimeoutError()
            calls += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n)) for n >= 2
        lam = 4 + int(3 * np.log(n)) if n >= 2 else 4
        # mu = lambda // 2 (half of the population for recombination)
        mu = lam // 2

        # Step sizes for different phases
        # Phase 1: Global exploration with large sigma
        # Phase 2: CMA-ES refinement

        # Multi-start: try several different starting points
        num_starts = 3
        calls_per_start = budget // num_starts

        for start_idx in range(num_starts):
            if time.time() > deadline or calls >= budget:
                break

            # Different starting points for diversity
            if start_idx == 0:
                x_mean = np.zeros(n)
                sigma = 3.0
            elif start_idx == 1:
                x_mean = rng.uniform(-3, 3, n)
                sigma = 2.0
            else:
                x_mean = rng.uniform(-4, 4, n)
                sigma = 1.5

            # Evaluate initial mean
            f_mean = tracked_func(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            # CMA-ES state
            C = np.eye(n)  # Covariance matrix
            p_sigma = np.zeros(n)  # Evolution path for sigma
            p_c = np.zeros(n)  # Evolution path for C
            c_sigma = 1 / (n + 4)
            d_sigma = 1 + 2 * max(0, (mu - 1) / (n + 2) - 1) + c_sigma
            c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * ((mu + 1.5 / n) - 2) / ((n + 2) ** 2 + mu))

            # Maximum number of iterations for this start
            max_iters = calls_per_start // lam
            iter_count = 0

            while iter_count < max_iters and time.time() < deadline and calls < budget:
                iter_count += 1

                # Generate population
                offspring = []
                fitnesses = []

                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Sample from N(x_mean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Apply Cholesky decomposition of C
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = tracked_func(y)
                    offspring.append(y)
                    fitnesses.append(fy)

                if len(offspring) < lam:
                    break

                # Sort by fitness
                indices = np.argsort(fitnesses)
                offspring = [offspring[i] for i in indices]
                fitnesses = [fitnesses[i] for i in indices]

                # Update best
                if fitnesses[0] < best_f:
                    best_f = fitnesses[0]
                    best_x = offspring[0].copy()

                # Recombination: weighted mean of the best mu offspring
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights ** 2)

                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * offspring[i]

                # Update evolution paths
                diff = (x_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                h_sigma = np.sqrt((1 - (1 - c_sigma) ** (iter_count + 1)) / (1 - (1 - c_sigma) ** (iter_count + 1)))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (iter_count + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)):
                    h_sigma = 1
                else:
                    h_sigma = 0

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Update covariance matrix
                C = ((1 - c_1 - c_mu) * C +
                     c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) +
                     c_mu / mu_eff * np.sum([np.outer(offspring[i] - x_mean, (offspring[i] - x_mean)) for i in range(mu)], axis=0) / mu_eff)

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size (CSA - Cumulative Step-size Adaptation)
                target = np.sqrt(n) * (1 + 2 * max(0, (mu - 1) / (n + 2) - 1))
                sigma *= np.exp((1 / d_sigma) * (np.linalg.norm(p_sigma) / target - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                x_mean = x_new

                # Check for convergence
                if sigma < 1e-8 or np.max(np.abs(x_mean)) > 50:
                    break

        # Final local refinement with (1+1)-ES around the best point
        if best_x is not None and time.time() < deadline - 2 and calls < budget:
            x = best_x.copy()
            fx = tracked_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 0.5
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = tracked_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
