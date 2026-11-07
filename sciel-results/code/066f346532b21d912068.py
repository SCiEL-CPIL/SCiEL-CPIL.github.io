"""Optimizer for BBOB-style f15 (Rastrigin rotated) with CMA-ES restarts and (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and (1+1)-ES polish for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, rng, x0, fx0, calls_made, best_f, best_x, deadline,
               budget, dim, lambda_pop, max_generations, sigma0):
        """Run CMA-ES from x0. Returns (best_x, best_f, calls_made)."""
        n = dim
        lam = lambda_pop
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_f_local = fx0
        best_x_local = x0.copy()
        gen = 0
        prev_best = fx0
        no_improve_count = 0

        for gen in range(max_generations):
            if calls_made >= budget or time.time() >= deadline:
                break

            # Generate lambda samples
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)

            y_samples = []
            x_samples = []
            f_samples = []

            for i in range(lam):
                if calls_made >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = B @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                calls_made += 1
                y_samples.append(y)
                x_samples.append(x_new)
                f_samples.append(f_new)

                if f_new < best_f_local:
                    best_f_local = f_new
                    best_x_local = x_new.copy()
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

            if len(f_samples) == 0:
                break

            # Sort by fitness
            order = np.argsort(f_samples)
            y_sorted = np.array([y_samples[i] for i in order])
            f_sorted = np.array([f_samples[i] for i in order])

            # Update mean
            y_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            x_mean = x_mean + sigma * y_mean
            x_mean = np.clip(x_mean, -5.0, 5.0)

            # Update covariance
            ys = y_sorted[:mu]
            diffmat = ys - y_mean[None, :]
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma) ** 2) * np.eye(n)) + c_mu * np.sum(weights[:, None, None] * diffmat[:, :, None] * diffmat[:, None, :], axis=0)

            # Update step size
            try:
                C_chol = np.linalg.cholesky(C)
                C_inv_chol = np.linalg.inv(C_chol)
                y_mean_norm = np.linalg.norm(y_mean)
                sign = 1 if np.dot(y_mean, C_chol @ C_inv_chol @ y_mean) > 0 else -1
                ps = (1 - c_sigma) * ps + c_sigma / chi_n * y_mean_norm * sign
            except np.linalg.LinAlgError:
                ps = (1 - c_sigma) * ps
                y_mean_norm = np.linalg.norm(y_mean)
                ps += c_sigma / chi_n * y_mean_norm

            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update pc
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            pc = (1 - c_c) * pc + h_sigma * c_c * y_mean

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2

            # Stopping criteria
            if sigma < 1e-8:
                break

            # Check for stagnation
            if f_sorted[0] >= prev_best - 1e-10:
                no_improve_count += 1
            else:
                no_improve_count = 0
                prev_best = f_sorted[0]

            if no_improve_count > 5:
                break

        return best_x_local, best_f_local, calls_made

    def _es_polish(self, func, rng, x0, fx0, calls_made, best_f, best_x, deadline, budget, dim, max_iters):
        """Polish with (1+1)-ES. Returns (best_x, best_f, calls_made)."""
        x = x0.copy()
        fx = fx0
        sigma = 0.3
        sigma_min = 1e-5
        sigma_max = 2.0
        success_window = []
        window_size = 10

        for _ in range(max_iters):
            if calls_made >= budget or time.time() >= deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                success_window.append(1)
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_window.append(0)

            if len(success_window) >= window_size:
                success_rate = sum(success_window[-window_size:]) / window_size
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                if len(success_window) > window_size * 2:
                    success_window = success_window[-window_size:]

            sigma = np.clip(sigma, sigma_min, sigma_max)

            if sigma < 1e-6:
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0

        # Phase 1: CMA-ES with multiple restarts
        # Use structured initial points that exploit Rastrigin's periodicity
        init_points = [
            np.zeros(dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([1.0, -1.0, 2.0, -2.0, 1.0]),
            rng.uniform(-4, 4, dim),
        ]

        lambda_pop = 10
        # Allocate budget: ~60% for CMA-ES restarts, ~30% for polish, ~10% reserve
        cma_budget = int(budget * 0.6)
        polish_budget = int(budget * 0.3)

        n_restarts = 3
        max_gens_per_restart = [
            max(5, (cma_budget // n_restarts) // lambda_pop),
            max(4, (cma_budget // (2 * n_restarts)) // lambda_pop),
            max(3, (cma_budget // (3 * n_restarts)) // lambda_pop),
        ]

        for r in range(n_restarts):
            if calls_made >= budget or time.time() >= deadline:
                break

            x0 = init_points[r].copy()
            x0 = np.clip(x0, -5.0, 5.0)
            fx0 = func(x0)
            calls_made += 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            if calls_made >= budget or time.time() >= deadline:
                break

            remaining = budget - calls_made
            max_gens = min(max_gens_per_restart[r], remaining // lambda_pop)
            if max_gens < 2:
                break

            sigma0 = 2.0 if r == 0 else 1.5

            best_x_local, best_f_local, calls_made = self._cmaes(
                func, rng, x0, fx0, calls_made, best_f, best_x,
                deadline, budget, dim, lambda_pop, max_gens, sigma0
            )

            if best_f_local < best_f:
                best_f = best_f_local
                best_x = best_x_local.copy()

        # Phase 2: (1+1)-ES polish from the best point found
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            max_polish_iters = min(remaining, polish_budget)

            best_x, best_f, calls_made = self._es_polish(
                func, rng, best_x, best_f, calls_made, best_f, best_x,
                deadline, budget, dim, max_polish_iters
            )

        # Phase 3: Final quick exploration if budget remains
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            # Try a few random perturbations around the best point
            for _ in range(min(remaining, 20)):
                if time.time() >= deadline or calls_made >= budget:
                    break
                y = best_x + rng.standard_normal(dim) * 0.5
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
