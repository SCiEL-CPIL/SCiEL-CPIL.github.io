"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        def safe_func(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            total_calls += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(7)

        def cma_es(x0, sigma0, budget_for_run, seed_offset=0):
            """Run CMA-ES with given initial point and step size."""
            nonlocal total_calls, best_x, best_f
            if not time_check() or total_calls >= max_calls:
                return

            rng_local = np.random.default_rng(42 + seed_offset)

            # CMA-ES parameters
            lam = max(4 * n, 8)  # population size
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = int(w.sum() ** 2 / w.sum())

            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - mueff / (n + 1) ** 1.5 + c_sigma) / ((n + 2) * (n / 2 + mueff - 1))
            c_mu = (mueff / (n + 2) - c_1) / (n / 2 + mueff - 1)
            c_mu = max(c_mu, 0)

            # State
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (20 * n * n))

            # Initial population
            g = 0
            while total_calls < max_calls and time_check():
                # Generate population
                pop = np.zeros((lam, n))
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    # Cholesky of C
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                        C = np.eye(n)
                    pop[i] = mean + sigma * (L @ z)

                # Evaluate
                fitness = np.zeros(lam)
                for i in range(lam):
                    if total_calls >= max_calls or not time_check():
                        fitness[i] = np.inf
                    else:
                        fitness[i] = safe_func(pop[i])

                if not np.isfinite(fitness).any():
                    break

                # Sort
                order = np.argsort(fitness)
                x_sorted = pop[order]
                f_sorted = fitness[order]

                # Update mean
                mean_new = (w[:mu] * x_sorted[:mu]).sum(axis=0)

                # Update sigma
                pc = (1 - c_sigma) * pc + c_sigma * (sigma / (chi_n * np.sqrt(c_sigma * (1 - c_sigma) * 1e-8 + 1e-30))) * (mean_new - mean)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (mean_new - mean) / max(sigma, 1e-10)

                if np.linalg.norm(ps) / chi_n < (1.4 + 2.0 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (g + 1))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                delta_sig = np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)) - 1
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (x_sorted[i] - mean) / max(sigma, 1e-10)
                    rank_mu += w[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Symmetrize
                C = (C + C.T) / 2

                mean = mean_new
                g += 1

                # Stopping: if sigma too small or no improvement in last few generations
                if sigma < 1e-10:
                    break

        # --- Phase 1: Quick random scan to find a good starting region ---
        # Evaluate a diverse set of points
        n_scan = min(30, max_calls // 5)
        scan_points = [np.zeros(n)]
        for _ in range(n_scan - 1):
            scan_points.append(rng.uniform(-4, 4, n))

        scan_fitness = []
        for p in scan_points:
            if total_calls >= max_calls or not time_check():
                break
            f = safe_func(p)
            scan_fitness.append(f)

        # Find the best scan point
        if scan_fitness:
            best_scan_idx = np.argmin(scan_fitness)
            start_point = scan_points[best_scan_idx].copy()
            start_sigma = 1.0
        else:
            start_point = np.zeros(n)
            start_sigma = 1.0

        # --- Phase 2: Multi-start CMA-ES ---
        # First, run CMA-ES from the best scan point
        remaining = max_calls - total_calls
        if remaining > 20:
            # Use 60% of remaining budget for first CMA-ES run
            budget_first = int(remaining * 0.5)
            cma_es(start_point, start_sigma, budget_first, seed_offset=0)

        # If we still have budget, try a few more starts from other good scan points
        if total_calls < max_calls and time_check() and scan_fitness:
            sorted_scan = np.argsort(scan_fitness)
            n_extra_starts = min(3, len(sorted_scan))
            for k in range(1, n_extra_starts + 1):
                if total_calls >= max_calls or not time_check():
                    break
                extra_point = scan_points[sorted_scan[k]].copy()
                # Only start if it's reasonably close to current best
                if best_x is not None:
                    dist = np.linalg.norm(extra_point - best_x)
                    if dist > 4.0:
                        continue
                remaining = max_calls - total_calls
                if remaining < 15:
                    break
                budget_extra = int(remaining * 0.3)
                cma_es(extra_point, 0.5, budget_extra, seed_offset=100 + k * 10)

        # --- Phase 3: Final local refinement with adaptive step ---
        if best_x is not None and total_calls < max_calls and time_check():
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.5
            stall_count = 0

            for _ in range(100):
                if total_calls >= max_calls or not time_check():
                    break

                improved = False
                for dim_idx in range(n):
                    for direction in [-1, 1]:
                        x_trial = x_curr.copy()
                        x_trial[dim_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        if np.isfinite(f_trial) and f_trial < f_curr - 1e-12:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break

                if improved:
                    step = min(step * 1.05, 2.0)
                else:
                    step *= 0.7
                    stall_count += 1
                    if step < 1e-4 or stall_count > 15:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            if total_calls < max_calls:
                best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
