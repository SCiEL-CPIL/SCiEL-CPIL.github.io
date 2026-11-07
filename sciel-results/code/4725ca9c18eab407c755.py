"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES with restarts and final local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget
        calls_made = 0

        global_best_x = None
        global_best_f = np.inf

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 4:
            lam = 4
        if lam > 12:
            lam = 12

        sigma0 = 1.5
        lb = -5.0
        ub = 5.0

        # Standard CMA-ES adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        def cma_es(x0, sigma_init, max_calls):
            nonlocal calls_made, global_best_x, global_best_f

            mean = x0.copy()
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Initial population
            pop = []
            for i in range(lam):
                if calls_made >= total_budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * z
                x = np.clip(x, lb, ub)
                fx = func(x)
                calls_made += 1
                pop.append((x, fx))
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()

            if len(pop) < lam:
                return

            # Main loop
            while calls_made < total_budget and time.time() < deadline - 0.5:
                # Sort population by fitness
                pop.sort(key=lambda t: t[1])

                # Update mean
                new_mean = np.sum([weights[i] * pop[i][0] for i in range(mu)], axis=0)

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C @ (new_mean - mean)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                # Update covariance
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * diff
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) - 1) * C)
                for i in range(mu):
                    x_i = (pop[i][0] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(x_i, x_i)

                mean = new_mean
                sigma = max(sigma, 1e-10)

                # Check for convergence
                if sigma < 1e-6:
                    break

                # Generate next population
                next_pop = []
                for i in range(lam):
                    if calls_made >= total_budget:
                        break
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, lb, ub)
                    fx = func(x)
                    calls_made += 1
                    next_pop.append((x, fx))
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()

                if len(next_pop) < lam:
                    break
                pop = next_pop

        # Strategy: Multiple restarts of CMA-ES
        # First restart gets a larger share to find a good basin
        # Subsequent restarts get smaller shares to refine or find other basins
        n_restarts = 3
        budgets = [int(total_budget * 0.5), int(total_budget * 0.3), int(total_budget * 0.2)]

        for i, b in enumerate(budgets):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break

            # Initialize restart
            if i == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma_init = 2.0
            else:
                # Restart near global best or random
                if global_best_x is not None and rng.random() < 0.5:
                    x0 = global_best_x + 2.0 * rng.standard_normal(dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                x0 = np.clip(x0, -5, 5)
                sigma_init = 1.0

            cma_es(x0, sigma_init, b)

        # Final polish with (1+1)-ES around global best
        if global_best_x is not None and calls_made < total_budget and time.time() < deadline - 1.0:
            x = global_best_x.copy()
            fx = global_best_f
            sigma = 0.5
            remaining = total_budget - calls_made
            refine_steps = min(remaining // 2, 100)

            for _ in range(refine_steps):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline - 0.5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 2.0)
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    sigma = max(sigma * 0.9, 0.01)

        # Return the best found
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
