"""Optimizer for BBOB f6 (Attractive sector) using global-to-local CMA-ES."""
import numpy as np
import time

class Optimizer:
    """Global-to-local CMA-ES optimizer for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling - 50 uniform points to identify best basin
        n_init = 50
        init_points = np.array([rng.uniform(-4.5, 4.5, dim) for _ in range(n_init)])
        init_fits = np.full(n_init, np.inf)

        for i in range(n_init):
            if time_left() < 10.0 or calls >= max_calls:
                break
            init_fits[i] = eval_func(init_points[i])

        if calls >= max_calls or time_left() < 15.0:
            return best_x, best_f

        # Phase 2: CMA-ES from the best initial point
        def cmaes(x0, sigma0, max_evals, seed_offset=0):
            nonlocal calls, best_f, best_x

            if max_evals < 20:
                return

            local_rng = np.random.default_rng(seed_offset)
            n = dim
            lambda_pop = 4 + int(3 * np.log(n))  # ~10 for dim=5
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x_mean = x0.copy()
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            def compute_sqrt_C(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

            sqrt_C = compute_sqrt_C(C)

            n_evals = 0
            stagnant_generations = 0
            best_in_cma = best_f
            best_x_in_cma = best_x.copy()
            best_f_hist = []

            while n_evals < max_evals and time_left() > 5.0:
                z = local_rng.standard_normal((lambda_pop, n))
                offsprings = x_mean + sigma * (z @ sqrt_C.T)

                fitness = np.full(lambda_pop, np.inf)
                for i in range(lambda_pop):
                    if n_evals >= max_evals or time_left() < 5.0:
                        break
                    fitness[i] = eval_func(offsprings[i])
                    n_evals += 1

                valid = np.isfinite(fitness)
                if np.sum(valid) < mu:
                    break

                indices = np.argsort(fitness)
                x_sorted = offsprings[indices]
                f_sorted = fitness[indices]

                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()

                if f_sorted[0] < best_in_cma:
                    best_in_cma = f_sorted[0]
                    best_x_in_cma = x_sorted[0].copy()
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

                best_f_hist.append(f_sorted[0])
                if len(best_f_hist) > 10:
                    best_f_hist.pop(0)

                # Update mean using reweighted selection
                x_mean_new = x_mean + sigma * sum(weights[i] * x_sorted[i] for i in range(mu))

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * sqrt_C.T @ (x_mean_new - x_mean)
                norm_ps = np.linalg.norm(ps)
                hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1) / lambda_pop))
                if hsig < (c1 + 1) * chi_n + 0.3:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma

                # Update covariance matrix - corrected formula
                diff = x_sorted[:mu] - x_mean
                C = (1 + c1 * (2 * cmu - 1) - c1) * C + c1 * np.outer(pc, pc) + cmu * (diff * weights[:, None]).T @ diff
                C = (C + C.T) / 2

                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals) + 1e-12) * np.eye(n)

                # Update step size
                sigma *= np.exp(cs / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e10)

                x_mean = x_mean_new
                sqrt_C = compute_sqrt_C(C)

                # Termination criteria
                if stagnant_generations > 15:
                    break
                if sigma < 1e-12:
                    break
                if len(best_f_hist) >= 10:
                    improvement = best_f_hist[0] - best_f_hist[-1]
                    if improvement < 1e-8 * max(1.0, abs(best_f_hist[-1])):
                        break

            return best_x_in_cma, best_in_cma

        # Run CMA-ES from the best initial point
        best_init_idx = np.argmin(init_fits)
        remaining = max_calls - calls
        cmaes(init_points[best_init_idx], 1.0, remaining, seed_offset=100)

        # Phase 3: If stagnation occurred, try restart from a diverse point
        remaining = max_calls - calls
        if remaining > 30 and time_left() > 10.0:
            # Find a diverse point from initial samples
            diverse_idx = -1
            min_dist = 0
            for i in range(n_init):
                if i == best_init_idx:
                    continue
                d = np.linalg.norm(init_points[i] - best_x)
                if d > min_dist:
                    min_dist = d
                    diverse_idx = i
            if diverse_idx >= 0 and min_dist > 2.0:
                cmaes(init_points[diverse_idx], 2.0, remaining, seed_offset=200)
            else:
                # Perturb the best point
                perturbed = best_x + rng.normal(0, 1.0, dim)
                perturbed = np.clip(perturbed, -4.5, 4.5)
                cmaes(perturbed, 1.0, remaining, seed_offset=300)

        return best_x, best_f
