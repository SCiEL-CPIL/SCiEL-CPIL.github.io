"""CMA-ES with adaptive restarts and local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local (1+1)-ES polish for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            nonlocal calls_made
            if time.time() > deadline:
                return True
            return False

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES with adaptive restarts
        def cmaes_run(func, max_calls, x0, sigma0, seed):
            nonlocal calls_made, best_x, best_f
            rng = np.random.default_rng(seed)
            n = dim
            # CMA-ES parameters
            mu = max(1, n // 2)
            lambda_ = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Initial population
            pop = np.zeros((lambda_, n))
            for i in range(lambda_):
                pop[i] = x_mean + sigma * B @ (D * rng.standard_normal(n))

            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                fvals[i] = call_func(np.clip(pop[i], -5, 5))
                if check_time():
                    return

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B @ (D * (x_mean - x_mean_old) / sigma)
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

            # Update covariance matrix
            c = (x_mean - x_mean_old) / sigma
            pc = (1 - c_c) * pc + c_c * c
            delta_diag = (1 - (1 - c_1) ** 2) * (1 + (1 - h_sigma) * c_mu / c_1) if h_sigma else (1 - (1 - c_1) ** 2)
            D **= 1 + 0.3 * (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
            B, D = np.linalg.eigh((1 - c_1 - c_mu + delta_diag) * np.outer(pc, pc) + c_1 * np.eye(n))
            D = 1.0 / np.sqrt(D)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Check convergence
            if np.linalg.norm(x_mean - x_mean_old) < 1e-12 * max(1, np.linalg.norm(x_mean)):
                return

            # Continue until budget or time
            while calls_made < max_calls and not check_time():
                # Generate new population
                for i in range(lambda_):
                    pop[i] = x_mean + sigma * B @ (D * rng.standard_normal(n))
                    fvals[i] = call_func(np.clip(pop[i], -5, 5))
                    if check_time():
                        return

                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B @ (D * (x_mean - x_mean_old) / sigma)
                h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

                c = (x_mean - x_mean_old) / sigma
                pc = (1 - c_c) * pc + c_c * c
                delta_diag = (1 - (1 - c_1) ** 2) * (1 + (1 - h_sigma) * c_mu / c_1) if h_sigma else (1 - (1 - c_1) ** 2)
                D **= 1 + 0.3 * (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
                B, D = np.linalg.eigh((1 - c_1 - c_mu + delta_diag) * np.outer(pc, pc) + c_1 * np.eye(n))
                D = 1.0 / np.sqrt(D)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                if np.linalg.norm(x_mean - x_mean_old) < 1e-12 * max(1, np.linalg.norm(x_mean)):
                    return

        # Phase 2: Local (1+1)-ES polish
        def local_polish(func, x_start, f_start, max_calls, seed):
            nonlocal calls_made, best_x, best_f
            rng = np.random.default_rng(seed)
            n = dim
            x = x_start.copy()
            fx = f_start
            sigma = 0.5

            while calls_made < max_calls and not check_time():
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                if sigma < 1e-10:
                    break

        # Main loop: multiple CMA-ES runs with different seeds, then polish
        seeds = [42, 123, 456, 789, 1024]
        budget_per_run = max(50, budget // len(seeds))

        for i, seed in enumerate(seeds):
            if calls_made >= budget or check_time():
                break
            # Start from a random point or best found so far
            if i == 0:
                x0 = np.random.default_rng(seed).uniform(-4, 4, dim)
            else:
                x0 = best_x + np.random.default_rng(seed).uniform(-1, 1, dim)
                x0 = np.clip(x0, -5, 5)

            cmaes_run(func, calls_made + budget_per_run, x0, 1.5, seed)

        # Final polish phase
        if calls_made < budget and not check_time() and best_x is not None:
            remaining = budget - calls_made
            local_polish(func, best_x, best_f, remaining, 999)

        return best_x, best_f
# EVOLVE-BLOCK-END
