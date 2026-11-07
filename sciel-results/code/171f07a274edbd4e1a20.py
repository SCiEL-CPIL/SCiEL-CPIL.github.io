"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based continuous optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Stage 1: Coarse random sampling to get an early decent value
        n_coarse = min(15, budget)
        for i in range(n_coarse):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_call(x)

        if calls >= budget or best_x is None:
            return best_x, best_f

        # Stage 2: CMA-ES with adaptive population and restarts
        x = best_x.copy()
        sigma = 1.0
        pop_size = max(4, int(4 + 3 * np.log(dim)))

        # Weights
        w = np.zeros(pop_size)
        for i in range(pop_size // 2):
            w[i] = np.log(pop_size / 2 + 0.5) - np.log(i + 1)
        w = w / w.sum()
        mu_eff = 1.0 / (w[:pop_size // 2] @ w[:pop_size // 2])

        # Parameters
        c_sigma = (pop_size + 2) / ((dim + 1.3) ** 2 + pop_size)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size // 2 + 1) - 2) / ((dim + 2) ** 2 + pop_size))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        no_improve = 0
        max_no_improve = int(pop_size * 15)
        restarts_done = 0
        max_restarts = 5

        while calls < budget:
            if time.time() > deadline:
                break

            # Check restart
            if no_improve >= max_no_improve:
                no_improve = 0
                restarts_done += 1
                if restarts_done > max_restarts:
                    sigma *= 0.5
                else:
                    if rng.random() < 0.3:
                        x = rng.uniform(-5.0, 5.0, dim)
                    else:
                        x = np.clip(x + 2.0 * rng.standard_normal(dim), -5.0, 5.0)
                    fx = safe_call(x)
                    sigma = max(sigma, 0.8)
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    continue

            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            population = []
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_call(y)
                population.append((y, fy))

            if len(population) == 0:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[1])

            # Update step size
            z_mean = np.zeros(dim)
            for i in range(min(pop_size // 2, len(population))):
                y, _ = population[i]
                z_i = np.linalg.solve(L, (y - x) / sigma)
                z_mean += w[i] * z_i

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update mean
            x_new = x + sum(w[i] * population[i][0] for i in range(min(pop_size // 2, len(population))))
            x_new = np.clip(x_new, -5.0, 5.0)

            # Update covariance
            y_vec = x_new - x
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec / sigma
            norm_ps = np.linalg.norm(ps)
            delta = (1 + (1 - c_c) * (1 - c_sigma) * (norm_ps ** 2 / chi_n - 1)) if norm_ps > (1.4 + 2 / (dim + 1)) * chi_n else 1.0
            C = (1 + c_1 * delta - c_1) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * sum(
                w[i] * np.outer((population[i][0] - x) / sigma, (population[i][0] - x) / sigma)
                for i in range(min(pop_size // 2, len(population)))
            )

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new

            # Check if we improved
            if best_f < np.inf:
                # We track improvement via best_f changes, but we need to detect no improvement
                # Let's track the best from this generation
                gen_best = population[0][1]
                if gen_best >= best_f - 1e-10:
                    no_improve += 1
                else:
                    no_improve = 0

        # Final polish
        if calls < budget and best_x is not None:
            for _ in range(min(10, budget - calls)):
                if time.time() > deadline:
                    break
                step = 0.05 * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5.0, 5.0)
                safe_call(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
