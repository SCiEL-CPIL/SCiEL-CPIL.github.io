# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based continuous optimizer with multi-start and adaptive restarts.
    
    Uses CMA-ES with:
    - Halton sequence for diverse initial points
    - Multiple CMA-ES runs with different starting points
    - Adaptive restarts when stagnation is detected
    - Boundary handling via reflection
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        # Generate diverse starting points using Halton sequence
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        n_starts = 8
        start_points = np.zeros((n_starts, dim))
        for i in range(1, n_starts + 1):
            for j in range(dim):
                base = primes[j % len(primes)]
                start_points[i - 1, j] = halton_sequence(i, base)
        start_points = start_points * 10.0 - 5.0

        # Add center point and random points
        start_points = np.vstack([
            start_points,
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim)
        ])

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Evaluate all starting points
        for pt in start_points:
            if calls >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal calls, best_f, best_x

            lambda_ = max(4 + int(3 * np.log(dim)), 8)
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0
            no_improve = 0
            local_best_f = best_f

            while calls < max_calls and time.time() < deadline:
                if sigma < 1e-10:
                    break

                # Generate population
                pop = np.zeros((lambda_, dim))
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    step = B @ (D * rng.standard_normal(dim))
                    x_candidate = x_mean + sigma * step
                    # Boundary handling: reflect
                    for d in range(dim):
                        if x_candidate[d] < -5:
                            x_candidate[d] = -5 + (-5 - x_candidate[d]) * 0.5
                        elif x_candidate[d] > 5:
                            x_candidate[d] = 5 - (x_candidate[d] - 5) * 0.5
                    x_candidate = np.clip(x_candidate, -5, 5)
                    pop[i] = x_candidate
                    fvals[i] = func(x_candidate)
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = x_candidate.copy()
                        no_improve = 0
                    elif fvals[i] < local_best_f:
                        local_best_f = fvals[i]
                        no_improve = 0
                    else:
                        no_improve += 1

                if calls >= max_calls:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Weighted mean
                x_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)

                # Update sigma (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * B @ (D * (x_new - x_mean))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)) / lambda_) / chi_n) < 1.4 + 2 / (dim + 1)
                ps *= hsig

                # Update pc (CMA)
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma

                # Update covariance matrix
                delta_mu = (1 - (1 - hsig) * c_c) * np.outer(pc, pc) / sigma ** 2 - np.eye(dim)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z = B @ (D * (pop_sorted[i] - x_mean) / sigma)
                    rank_mu += weights[i] * np.outer(z, z)

                C = (1 + c_1 * delta_mu + c_mu * rank_mu) * (B @ np.diag(D ** 2) @ B.T)

                # Eigen-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update mean
                x_mean = x_new

                # Adapt sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                gen += 1

                # Stagnation check
                if no_improve > lambda_ * 3:
                    break

            return best_f, best_x, calls

        # Run multiple CMA-ES instances from different starting points
        remaining_budget = budget - calls
        n_runs = 4
        budget_per_run = remaining_budget // n_runs

        for i in range(n_runs):
            if calls >= budget or time.time() > deadline:
                break

            # Choose starting point: best found so far for first run, others from Halton
            if i == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                x0 = start_points[i % len(start_points)].copy()
                sigma0 = 2.0

            # Run CMA-ES
            local_best_f, local_best_x, calls = run_cmaes(x0, sigma0, calls + budget_per_run)

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Final refinement: short CMA-ES from best point with small sigma
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                local_best_f, local_best_x, calls = run_cmaes(best_x.copy(), 0.1, calls + remaining)
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
