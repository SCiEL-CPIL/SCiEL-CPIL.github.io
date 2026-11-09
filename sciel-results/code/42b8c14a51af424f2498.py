"""Black-box continuous optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Büche-Rastrigin (BBOB f4) using multi-start CMA-ES
    with Halton-based initialization for diverse starting points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or check_deadline():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Generate diverse starting points using Halton sequence ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton_points(n, dim):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11]
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, primes[j])
            return points

        # Generate 20 Halton points for initial exploration
        n_halton = 20
        halton_points = generate_halton_points(n_halton, dim)
        halton_points = halton_points * 10.0 - 5.0

        # Add origin and a few random points
        extra_points = [np.zeros(dim)]
        extra_points.extend(rng.uniform(-4, 4, size=(5, dim)))

        all_starts = np.vstack([halton_points, np.array(extra_points)])
        rng.shuffle(all_starts)

        # Evaluate all starting points
        starts = []
        for x in all_starts:
            fx = evaluate(x)
            if fx is None:
                break
            starts.append((fx, x.copy()))

        if best_x is None:
            return None, np.inf

        # Sort by function value
        starts.sort(key=lambda t: t[0])

        # --- Phase 2: Multi-start CMA-ES ---
        def cma_es(x0, max_calls, sigma0=1.0):
            nonlocal calls_made, best_f, best_x, deadline

            if check_deadline() or calls_made >= budget:
                return

            # CMA-ES parameters
            n = dim
            # Population size: lambda = 4 + floor(3 * ln(n))
            lam = 4 + int(3 * np.log(n + 1))
            # Adaptation parameters
            cmu = 1 / (5 * n + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mu))
            cs = (mu + 2) / (n + mu + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            mu = lam // 2  # Number of best solutions used for update

            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            c_sigma_inv = 1.0 / ds
            c_c_inv = 1.0 / cmu
            c_1_inv = 1.0 / c1
            c_c_inv = 1.0 / cmu

            # Precompute weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c1 = (2 / ((n + 1.3) ** 2 + mu_eff))
            cmu = min(1 - c1, (mu_eff - 2) / (n + 2))

            best_local_f = np.inf
            best_local_x = x0.copy()

            while calls_made < budget and not check_deadline():
                # Generate population
                population = []
                for i in range(lam):
                    if check_deadline() or calls_made >= budget:
                        return
                    z = rng.standard_normal(n)
                    # Use Cholesky decomposition for C
                    L = np.linalg.cholesky(C)
                    y = L @ z
                    xi = x_mean + sigma * y
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = evaluate(xi)
                    if fi is None:
                        return
                    population.append((fi, xi.copy()))

                # Sort by fitness
                population.sort(key=lambda t: t[0])

                # Update best
                for fi, xi in population[:mu]:
                    if fi < best_local_f:
                        best_local_f = fi
                        best_local_x = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                # Compute new mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * population[i][1]
                x_mean = new_mean
                x_mean = np.clip(x_mean, -5.0, 5.0)

                # Update evolution paths
                for i in range(mu):
                    y_i = (population[i][1] - x_mean) / sigma
                    pc = (1 - cs) * pc + cs * np.sum(weights[i] * y_i for i in range(mu))
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (x_mean - x_mean) / sigma

                # Update covariance matrix
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (population[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * (rank_one + (1 + cs - c1) * np.eye(n) - C) + cmu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-8, min(sigma, 10.0))

                # Check for convergence
                if np.linalg.norm(x_mean - population[0][1]) < 1e-6 and sigma < 1e-4:
                    break

        # Allocate budget among top starting points
        remaining = budget - calls_made
        n_starts_to_optimize = min(5, len(starts))
        if n_starts_to_optimize == 0:
            return best_x, best_f

        # Give more budget to better starting points
        # Weights: 40%, 25%, 15%, 10%, 10%
        weight_list = [0.4, 0.25, 0.15, 0.1, 0.1]
        if n_starts_to_optimize < 5:
            weight_list = [1.0 / n_starts_to_optimize] * n_starts_to_optimize

        for i in range(n_starts_to_optimize):
            if check_deadline() or calls_made >= budget:
                break

            fx, x0 = starts[i]
            # Estimate sigma based on function landscape
            # Büche-Rastrigin has local minima roughly 2 units apart
            sigma0 = 1.0

            alloc = int(remaining * weight_list[i])
            if i == n_starts_to_optimize - 1:
                alloc = budget - calls_made

            if alloc > 10:
                cma_es(x0, alloc, sigma0)

        # --- Phase 3: Final fine-tuning with small random walk ---
        if calls_made < budget and best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            stall_count = 0
            max_stall = 50

            while calls_made < budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.05, 0.5)
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.95
                    stall_count += 1

                if stall_count > max_stall:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
