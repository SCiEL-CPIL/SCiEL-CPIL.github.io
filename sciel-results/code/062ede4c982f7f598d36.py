"""CMA-ES based multi-start optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES multi-start optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(42)

        def cma_es(x0, sigma0, max_evals):
            """Run a simplified CMA-ES from x0 with limited evaluation budget."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size adaptation parameters
            cs = (mueff + 1) / (n + mueff + 3)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            # Covariance matrix adaptation
            c1 = (2 - mueff / (n + 2) ** 1.5 + 2 * weights[0]) / ((n + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State
            xmean = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(n)
            Cinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            evals_this_run = 0

            while evals_this_run < max_evals and calls < budget and time.time() < deadline:
                if time.time() > deadline - 5.0:
                    break

                # Generate population
                population = []
                for i in range(lambda_):
                    # Sample from N(xmean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Cholesky decomposition for C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    xi = xmean + sigma * L @ z
                    xi = np.clip(xi, -5.0, 5.0)
                    population.append(xi)

                # Evaluate population
                fitness = []
                for xi in population:
                    if calls >= budget or time.time() > deadline:
                        break
                    f_val = eval_and_track(xi)
                    fitness.append(f_val)

                if len(fitness) < 2:
                    break

                # Sort by fitness
                indices = np.argsort(fitness)
                sorted_fitness = np.array(fitness)[indices]
                sorted_population = [population[i] for i in indices]

                # Update best
                if sorted_fitness[0] < best_f:
                    best_f = sorted_fitness[0]
                    best_x = sorted_population[0].copy()

                # Update mean
                new_mean = np.sum([weights[i] * sorted_population[i] for i in range(mu)], axis=0)

                # Update evolution paths
                try:
                    Cinv_sqrt = np.linalg.cholesky(Cinv)
                    Cinv_sqrt_inv = np.linalg.inv(Cinv_sqrt)
                except np.linalg.LinAlgError:
                    Cinv_sqrt_inv = np.eye(n)

                y = (new_mean - xmean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt_inv @ y
                norm_ps = np.linalg.norm(ps)
                hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (evals_this_run + 1)) / chi_n < 1.4 + 2 / (n + 1))
                pc = (1 - cs) * pc + hsig * (cs / (1 - cs)) * y

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                outer_y = np.outer(y, y)
                outer_pc = np.outer(pc, pc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (outer_pc + delta_hsig * C) + cmu / sigma ** 2 * np.sum(
                    [weights[i] ** 2 * np.outer(sorted_population[i] - xmean, sorted_population[i] - xmean) for i in range(mu)],
                    axis=0
                )

                # Ensure symmetry
                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update mean
                xmean = new_mean
                xmean = np.clip(xmean, -5.0, 5.0)

                # Update Cinv
                try:
                    Cinv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    Cinv = np.eye(n)

                evals_this_run += lambda_

        # Phase 1: Coarse random search to find promising basins
        n_random = max(15, budget // 4)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES refinement on top candidates
        top_k = min(4, len(candidates))

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x0 = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20:
                break

            # Initial sigma: distance to next best point or default
            if i < len(candidates) - 1:
                sigma0 = max(0.5, np.linalg.norm(x0 - candidates[i + 1][1]))
            else:
                sigma0 = 1.0

            cma_es(x0, sigma0, remaining)

        # Phase 3: If budget remains, do local random search around best
        if calls < budget and time.time() < deadline:
            step = 0.5
            while calls < budget and time.time() < deadline:
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    step = max(0.1, step * 0.85)
                else:
                    step = min(2.0, step * 1.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
