"""CMA-ES optimizer for BBOB f6 (Attractive Sector) with restarts and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts, structured init, and adaptive local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Lightweight structured initialization
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        # Quick axis probes (smaller step to save budget)
        probe_len = 2.5
        for i in range(dim):
            for sign in [1, -1]:
                if check_deadline() or calls >= budget:
                    break
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # A few random probes to find a better starting point
        n_random = min(10, max(0, budget - calls - 20))
        for _ in range(n_random):
            if check_deadline() or calls >= budget:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with stagnation detection and restarts
        lam = max(4, int(4 + 3 * np.log(max(dim, 2))))  # ~10 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x_start, sigma_start, remaining_budget):
            nonlocal calls
            if remaining_budget < lam:
                return x_start, best_f, calls

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            C_inv = np.eye(dim)
            try:
                C_chol = np.linalg.cholesky(C)
            except:
                C_chol = np.eye(dim)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            best_x_local = x_start.copy()
            best_f_local = func(x_start)
            calls += 1
            remaining_budget -= 1

            no_improve = 0
            best_f_at_last_improve = best_f_local

            for iteration in range(remaining_budget // lam + 1):
                if check_deadline() or calls >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    if check_deadline() or calls >= budget:
                        break
                    z = rng.standard_normal(dim)
                    try:
                        y = x_mean + sigma * (C_chol @ z)
                    except:
                        y = x_mean + sigma * z
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    offspring.append((y, fy))

                if len(offspring) < mu:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                selected = offspring[:mu]

                # Track best
                if selected[0][1] < best_f_local:
                    best_f_local = selected[0][1]
                    best_x_local = selected[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Stagnation check: restart if no improvement for too many generations
                if no_improve > 3 * dim and iteration > 5:
                    break

                # Weighted mean of selected
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * selected[i][0]

                # Update paths
                try:
                    C_inv_sqrt = np.linalg.cholesky(C_inv)
                    C_inv_sqrt_T = C_inv_sqrt.T
                except:
                    C_inv_sqrt_T = np.eye(dim)

                diff = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)

                p_sigma_norm = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Update covariance (FIXED: correct formula without trailing *C)
                delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
                rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)

                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = selected[i][0] - x_mean
                    rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

                C = C + c_1 * (rank_one - C) + c_mu * (rank_mu - C)
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C_inv and C_chol
                x_mean = x_new_mean
                try:
                    C_inv = np.linalg.inv(C)
                    C_chol = np.linalg.cholesky(C)
                except:
                    # Reset if covariance becomes singular
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_chol = np.eye(dim)
                    sigma = max(sigma, 0.5)

            return best_x_local, best_f_local, calls

        # Run CMA-ES with potential restarts
        remaining = budget - calls
        # First CMA-ES run with moderate initial sigma
        x_result, f_result, calls = run_cmaes(best_x.copy(), 1.0, remaining)

        if f_result < best_f:
            best_f = f_result
            best_x = x_result

        # If we still have budget and haven't converged, try a restart from best
        if not check_deadline() and calls < budget:
            remaining2 = budget - calls
            if remaining2 >= lam:
                # Restart with smaller sigma for refinement
                x_result2, f_result2, calls = run_cmaes(best_x.copy(), 0.3, remaining2)
                if f_result2 < best_f:
                    best_f = f_result2
                    best_x = x_result2

        # Phase 3: Adaptive local (1+1)-ES polish
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.1
            successes = 0
            total = 0
            min_sigma = 1e-6
            max_sigma = 1.0

            n_local = budget - calls
            for _ in range(n_local):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total += 1
                if total >= 4:
                    sr = successes / total
                    if sr > 0.3:
                        sigma_local = min(sigma_local * 1.3, max_sigma)
                    elif sr < 0.1:
                        sigma_local = max(sigma_local * 0.7, min_sigma)
                    successes = 0
                    total = 0

            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
