"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls_used = 0

        # Helper to check time
        def time_exceeded():
            return time.time() >= deadline

        # Phase 1: Multi-start CMA-ES
        # We'll do multiple restarts of CMA-ES with different initial points
        # and population sizes, keeping the best found.

        def run_cmaes(x0, sigma0, budget_cma, rng, func, call_counter):
            """Run CMA-ES starting from x0 with initial sigma."""
            n = self.dim
            # CMA-ES parameters
            mu = max(2, n // 2)  # number of parents
            lambda_pop = 2 * n  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / (n ** 1.7 + 10)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n ** 1.7 + 12))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            best_x_local = x_mean.copy()
            best_f_local = func(x_mean)
            call_counter[0] += 1

            for gen in range(budget_cma):
                if time_exceeded():
                    break

                # Generate population
                for i in range(lambda_pop):
                    if time_exceeded():
                        break
                    z = rng.standard_normal(n)
                    x_new = x_mean + sigma * (z @ C)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    call_counter[0] += 1

                    if f_new < best_f_local:
                        best_f_local = f_new
                        best_x_local = x_new.copy()

                # Sort by fitness
                # We need to store all evaluations for this generation
                # Let's restructure: collect all, then sort
                break  # This simple loop doesn't work well; let's use a proper CMA-ES

            return best_x_local, best_f_local, call_counter[0]

        # Let's implement a proper CMA-ES
        def cmaes(x0, sigma0, budget_cma, rng, func):
            n = self.dim
            mu = max(2, n // 2)
            lambda_pop = 2 * n
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / (n ** 1.7 + 10)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (n ** 1.7 + 12))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            best_x_local = x0.copy()
            best_f_local = func(x0)
            calls = 1

            for gen in range(budget_cma):
                if time.time() >= deadline:
                    break

                # Generate and evaluate population
                pop = []
                for i in range(lambda_pop):
                    if time.time() >= deadline:
                        break
                    z = rng.standard_normal(n)
                    x_new = x_mean + sigma * (z @ C)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    pop.append((f_new, x_new))

                if len(pop) == 0:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[0])

                # Update best
                if pop[0][0] < best_f_local:
                    best_f_local = pop[0][0]
                    best_x_local = pop[0][1].copy()

                # Update mean
                y_w = np.zeros(n)
                for i in range(min(mu, len(pop))):
                    y_w += weights[i] * pop[i][1]
                y_w /= sigma

                # Update path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_w
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

                # Update covariance
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n < (1 + 2 / (n + 1)) * c_sigma)
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) - C) + \
                    c_mu * sum(weights[i] * np.outer(pop[i][1] - x_mean, pop[i][1] - x_mean) for i in range(min(mu, len(pop)))) / sigma ** 2

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update mean
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5, 5)

            return best_x_local, best_f_local, calls

        # Main loop: multiple restarts
        rng = np.random.default_rng(42)
        call_counter = [0]

        # Initial exploration: try a few random points
        n_initial = min(10, self.budget)
        for i in range(n_initial):
            if time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            call_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES restarts
        n_restarts = 5
        budget_per_restart = (self.budget - call_counter[0] - 50) // n_restarts
        if budget_per_restart < 50:
            budget_per_restart = 50
            n_restarts = max(1, (self.budget - call_counter[0] - 50) // budget_per_restart)

        for r in range(n_restarts):
            if time.time() >= deadline:
                break
            if call_counter[0] >= self.budget - 10:
                break

            # Start from best found so far with some perturbation
            if best_x is not None:
                x0 = best_x + rng.standard_normal(self.dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            else:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 2.0

            # Adjust budget for this restart
            remaining = self.budget - call_counter[0] - 10
            b_cma = min(budget_per_restart, remaining)
            if b_cma < 20:
                break

            x_best, f_best, calls_used_cma = cmaes(x0, sigma0, b_cma, rng, func)
            call_counter[0] += calls_used_cma

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 2: Polish with (1+1)-ES if time and budget allow
        if best_x is not None and call_counter[0] < self.budget - 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(50, self.budget - call_counter[0] - 5)):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                call_counter[0] += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
