# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with adaptive restarts and Halton-based initialization
    for BBOB-style optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Helper: Halton sequence
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def generate_halton(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    points[i - 1, j] = halton_sequence(i, base)
            return points

        # Global state
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Halton-based coarse initialization
        n_init = 30
        halton_init = generate_halton(n_init, dim)
        halton_init = halton_init * 10.0 - 5.0

        for pt in halton_init:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: CMA-ES with adaptive restarts
        def cma_es(x0, sigma0, remaining_budget, deadline):
            nonlocal best_x, best_f, calls

            dim = self.dim
            n = max(4, int(round(1 + 3 * np.log(dim))))  # population size
            if n > remaining_budget // 5:
                n = max(4, remaining_budget // 5)

            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)

            # CMA-ES parameters
            mu = n // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1) / ((dim + 2) ** 2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            ps_norm = 1.0 / np.sqrt(1 - (1 - c_sigma) ** 2)

            # Cholesky helper
            def chol_update(C, C_sqrt_inv, c_1, c_mu, weights, y_w, dim):
                # Update C
                C += (c_1 + c_mu) * np.outer(y_w, y_w) - c_1 * C
                # Re-orthogonalize
                C = (C + C.T) / 2
                # Eigenvalue decomposition to ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # Update C_sqrt_inv via eigenvalue decomposition
                eigvals_inv = 1.0 / np.sqrt(eigvals)
                C_sqrt_inv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
                return C, C_sqrt_inv

            iterations = 0
            stagnation = 0
            no_improve = 0

            while calls < budget and iterations < remaining_budget:
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(n):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5, 5)
                    population.append(y)

                # Evaluate population
                fitness = []
                for y in population:
                    fy = func(y)
                    calls += 1
                    fitness.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                if calls >= budget:
                    break

                # Sort by fitness
                order = np.argsort(fitness)
                population_sorted = [population[i] for i in order]
                fitness_sorted = [fitness[i] for i in order]

                # Check for improvement
                if fitness_sorted[0] < best_f:
                    best_f = fitness_sorted[0]
                    best_x = population_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Weighted mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * population_sorted[i]
                y_w -= x

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_w) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / ps_norm - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                if np.linalg.norm(y_w) > 1.4 + 0.3 / dim * np.sqrt(mu_eff) * ps_norm:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
                C, C_sqrt_inv = chol_update(C, C_sqrt_inv, c_1, c_mu, weights, y_w, dim)

                # Update mean
                x = y_w + x
                x = np.clip(x, -5, 5)

                iterations += 1

                # Restart conditions
                if no_improve > 100:
                    break
                if np.linalg.norm(x - best_x) < 1e-6 and no_improve > 20:
                    break
                if sigma < 1e-8:
                    break

            return best_x, best_f, calls

        # Run CMA-ES with restarts
        sigma0 = 1.0
        n_restarts = 5
        restarts_done = 0

        while calls < budget and restarts_done < n_restarts:
            if time.time() > deadline:
                break

            remaining = budget - calls
            if remaining < 10:
                break

            # Try CMA-ES from current best
            x_start = best_x.copy()
            sigma_start = 1.0 if restarts_done == 0 else 0.5

            # Run CMA-ES
            best_x, best_f, calls = cma_es(x_start, sigma_start, remaining, deadline)

            if time.time() > deadline:
                break

            if calls >= budget:
                break

            restarts_done += 1

            # If we're still improving, continue with smaller sigma
            if best_f > -1000:
                # Try a local refinement
                n_local = 10
                local_points = best_x + 0.1 * rng.standard_normal((n_local, dim))
                local_points = np.clip(local_points, -5, 5)
                for pt in local_points:
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    f = func(pt)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = pt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
