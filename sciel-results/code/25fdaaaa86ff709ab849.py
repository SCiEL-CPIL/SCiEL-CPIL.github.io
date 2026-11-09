import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Initial sampling to find promising basins
        n_initial = min(20, budget)
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_initial):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness, keep top few as potential starting points
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        starting_points = [c[1] for c in candidates[:top_k]]

        # Phase 2: CMA-ES optimization from best starting point
        # CMA-ES implementation
        def cma_es(x0, sigma0, remaining_budget, max_iter=None):
            """Simple CMA-ES implementation."""
            n = len(x0)
            if max_iter is None:
                max_iter = remaining_budget

            # CMA-ES parameters
            lambda_ = int(4 + 3 * np.log(n))  # population size
            mu = lambda_ // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            # Step size control
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            calls = 0

            best_f_local = np.inf
            best_x_local = x0.copy()

            for gen in range(max_iter):
                if not time_ok():
                    break
                if calls >= remaining_budget:
                    break

                # Generate lambda_ offspring
                offspring = []
                for i in range(lambda_):
                    if not time_ok() or calls >= remaining_budget:
                        break
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((fy, y.copy()))
                    if fy < best_f_local:
                        best_f_local = fy
                        best_x_local = y.copy()

                if len(offspring) < mu:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                f_selected = np.array([s[0] for s in selected])
                x_selected = np.array([s[1] for s in selected])

                # Update mean
                x_new_mean = np.sum(weights[:, None] * x_selected, axis=0)

                # Update covariance
                diff = (x_selected - x_mean) / sigma
                delta_mean = x_new_mean - x_mean
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(delta_mean / sigma, delta_mean / sigma) + C) + \
                    c_mu * np.sum(weights[:, None] * np.outer(diff, diff), axis=0)

                # Symmetrize
                C = 0.5 * (C + C.T)

                # Update path
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) / np.linalg.norm(np.linalg.cholesky(C)) * \
                     (x_new_mean - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / \
                     np.linalg.cholesky(C) * (x_new_mean - x_mean) / sigma

                # Step size adaptation
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)

                # Stagnation check: if no improvement for many generations, break
                if gen > 0 and f_selected[0] >= best_f_local + 1e-10 and gen % 5 == 0:
                    # Check if we've been stuck
                    pass

            return best_x_local, best_f_local, calls

        # Run CMA-ES from the best initial point
        remaining = budget - n_initial
        if remaining > 5:
            x_best, f_best, used = cma_es(best_x.copy(), 2.0, remaining)
            n_initial += used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # Try CMA-ES from second best starting point if we have budget
            if len(starting_points) > 1 and budget - n_initial > 20:
                x2, f2, used2 = cma_es(starting_points[1].copy(), 1.5, budget - n_initial)
                n_initial += used2
                if f2 < best_f:
                    best_f = f2
                    best_x = x2

        # Phase 3: Local refinement with small random perturbations
        if time_ok() and n_initial < budget:
            sigma_local = 0.1
            for _ in range(budget - n_initial):
                if not time_ok():
                    break
                x_try = best_x + sigma_local * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                fx_try = func(x_try)
                n_initial += 1
                if fx_try < best_f:
                    best_f = fx_try
                    best_x = x_try
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.9
                sigma_local = np.clip(sigma_local, 0.01, 5.0)

        return best_x, best_f
