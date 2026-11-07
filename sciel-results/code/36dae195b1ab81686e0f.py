"""Rewritten optimizer for BBOB f15 (Rastrigin rotated)."""
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: coarse global grid + multi-basin CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline

        # Phase 1: Coarse global exploration with stratified sampling
        # Use a grid-based approach with some jitter for better coverage
        n_per_dim = 3  # 3^5 = 243 points
        # But budget is only 500, so use a smarter approach
        # Use 2^5 = 32 corner/center points plus some random for diversity
        
        # Actually, let's use a Halton sequence for quasi-random coverage
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                base = primes[d]
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        i_val, rem = divmod(i_val, base)
                        f /= base
                        points[i, d] += rem * f
            return points

        n_global = 100
        halton = halton_sequence(n_global, dim)
        halton = halton * 10.0 - 5.0  # Scale to [-5, 5]

        rng = np.random.default_rng(12345)
        # Add some random points for diversity
        n_random = 50
        random_pts = rng.uniform(-5, 5, size=(n_random, dim))
        
        all_global_pts = np.vstack([halton, random_pts])
        n_global_total = len(all_global_pts)

        best_f = np.inf
        best_x = None
        candidates = []

        for i in range(n_global_total):
            if not time_ok():
                break
            x = np.clip(all_global_pts[i], -5, 5)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort and select top 5 diverse basins
        candidates.sort(key=lambda t: t[0])
        
        # Select top 5 with diversity constraint
        selected_basins = []
        min_dist = 1.5
        for fx, x in candidates:
            if len(selected_basins) >= 5:
                break
            # Check diversity
            diverse = True
            for _, sx in selected_basins:
                if np.linalg.norm(x - sx) < min_dist:
                    diverse = False
                    break
            if diverse:
                selected_basins.append((fx, x.copy()))
        
        # If we didn't get 5 diverse, fill with top remaining
        if len(selected_basins) < 5:
            for fx, x in candidates:
                if len(selected_basins) >= 5:
                    break
                if (fx, x) not in selected_basins:
                    selected_basins.append((fx, x.copy()))

        calls_used = n_global_total

        # Phase 2: CMA-ES on each basin with allocated budget
        n_basins = len(selected_basins)
        budget_per_basin = max(20, (budget - calls_used) // max(1, n_basins))

        def cmaes(func, x0, f0, budget_cma, dim, deadline, start_time, rng):
            """Run CMA-ES from x0 with f0 for budget_cma evaluations."""
            if budget_cma < 5:
                return x0, f0

            pop_size = max(4, int(4 + 3 * np.log(dim)))
            if pop_size > budget_cma // 2:
                pop_size = max(4, budget_cma // 2)

            x_mean = x0.copy()
            f_best = f0
            x_best = x0.copy()
            sigma = 1.0

            C = np.eye(dim)
            L = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            c_sigma = (pop_size + 2) / (dim + 4)
            c_c = (4 + dim / 4) / (dim + 4 + pop_size)
            mu = max(1, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_1 = (2 / ((dim + 1.3) ** 2 + pop_size)) / mu_eff
            c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + pop_size)) / mu_eff

            calls = 0
            gen = 0

            while calls < budget_cma and time.time() - start_time < deadline:
                gen += 1
                offspring = []

                for i in range(pop_size):
                    if calls >= budget_cma or time.time() - start_time >= deadline:
                        break

                    z = rng.standard_normal(dim)
                    try:
                        x_child = x_mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        x_child = x_mean + sigma * z

                    x_child = np.clip(x_child, -5, 5)
                    f_child = func(x_child)
                    calls += 1
                    offspring.append((f_child, x_child))

                    if f_child < f_best:
                        f_best = f_child
                        x_best = x_child.copy()

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]

                # Update mean
                new_mean = np.zeros(dim)
                for w, (_, x) in zip(weights, selected):
                    new_mean += w * x

                # Update p_sigma
                d_sigma = np.sqrt((1 + 2 / (dim + 1)) * c_sigma * (1 - c_sigma))
                for _, x in selected:
                    diff = (x - x_mean) / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff / d_sigma

                # h_sigma
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * h_sigma * (new_mean - x_mean) / sigma

                # Update C
                C = ((1 + c_1 * (2 - c_1) - c_mu) * C
                     + c_1 * np.outer(p_c, p_c)
                     + c_mu * np.sum([w * np.outer((x - x_mean) / sigma, (x - x_mean) / sigma)
                                       for w, (_, x) in zip(weights, selected)], axis=0))

                # Ensure positive semi-definite
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(1e-8, min(sigma, 10.0))

                x_mean = new_mean

            return x_best, f_best

        # Run CMA-ES on each basin
        for basin_fx, basin_x in selected_basins:
            if not time_ok():
                break
            x_opt, f_opt = cmaes(func, basin_x, basin_fx, budget_per_basin, dim, deadline, start_time, rng)
            calls_used += budget_per_basin
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: Final polish with (1+1)-ES
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            sigma_polish = 0.3
            x_cur = best_x.copy()
            f_cur = best_f
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x_cur + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f_cur:
                    x_cur, f_cur = y, fy
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.95
            if f_cur < best_f:
                best_f = f_cur
                best_x = x_cur.copy()

        return best_x, best_f
