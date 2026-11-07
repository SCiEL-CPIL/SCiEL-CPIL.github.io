"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard, multi-start, and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 270.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls_used
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Multi-start with different initializations
        # Use a few starting points to handle the hidden shift/rotation
        # Start with center, then a few random points

        # Phase 1: CMA-ES from multiple starts
        # We'll do CMA-ES with a population, restarting if stuck

        def run_cmaes(x_start, sigma_init, max_calls):
            nonlocal calls_used, best_x, best_f
            local_calls = 0
            x_mean = x_start.copy()
            sigma = sigma_init

            lam = 4 + int(3 * np.log(dim))  # ~11 for dim=5
            mu = lam // 2

            # Early stopping: if sigma becomes very small, we're converged
            min_sigma = 1e-10

            cmac = 1.0 / (dim + 4)
            cmam = (4 + dim / 4) / (dim + 4 + 3)
            c_sigma = (dim + 1.3) / (dim + 5.3 + 2 * lam / mu)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            no_improve = 0
            best_local_f = np.inf

            while local_calls < max_calls:
                if time.time() > deadline - 10.0:
                    break
                if calls_used >= budget:
                    break

                # Generate population
                offspring = np.zeros((lam, dim))
                fitness = np.zeros(lam)

                for i in range(lam):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 10.0:
                        break

                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring[i] = y
                    f_val = func(y)
                    calls_used += 1
                    local_calls += 1
                    fitness[i] = f_val

                    if f_val < best_local_f:
                        best_local_f = f_val
                        no_improve = 0
                    update_best(y, f_val)

                if calls_used >= budget or time.time() > deadline - 10.0:
                    break

                # Sort by fitness
                idx = np.argsort(fitness)
                offspring = offspring[idx]
                fitness = fitness[idx]

                # Weighted recombination
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += w[i] * offspring[i]

                # Compute v BEFORE updating mean
                v = (x_new - x_mean) / sigma

                # Update mean
                x_mean = x_new
                pc = (1 - cmac) * pc + cmac * v
                norm_pc = np.linalg.norm(pc)
                hsig = (norm_pc / np.sqrt(1 - (1 - cmac) ** 2) / chi_n <
                        (1.4 + 2 / (dim + 1)) * c_sigma)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * v / chi_n

                delta_h = (1 - hsig) * c_sigma * (2 - c_sigma)
                C = ((1 - cmac * (2 - cmac) + cmac * cmam * delta_h) * C
                     + (cmac * (2 - cmac) - cmam * delta_h) * np.outer(pc, pc)
                     + c_sigma * (2 - c_sigma) * mueff * np.outer(ps, ps))

                C = (C + C.T) / 2
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Check for stagnation
                if fitness[0] - fitness[-1] < 1e-12:
                    no_improve += 1
                else:
                    no_improve = 0

                if no_improve > 30 or sigma < min_sigma:
                    break

            return local_calls

        # Multi-start: try center and a few random points
        starts = [np.zeros(dim)]
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, size=dim))

        # Allocate budget: main CMA-ES gets most, then polish
        # First start gets the bulk of the budget
        cmaes_budgets = [300, 100, 50]
        polish_budget = 50

        for i, start_x in enumerate(starts):
            if calls_used >= budget:
                break
            if time.time() > deadline - 20.0:
                break

            # Run CMA-ES from this start
            run_cmaes(start_x, 2.0, cmaes_budgets[i])

        # Phase 2: (1+1)-ES polish from best point found
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            # Main polish: adapt sigma with (1+1)-ES strategy
            polish_calls = 0
            max_polish = min(polish_budget, budget - calls_used)
            for _ in range(max_polish):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    update_best(x, fx)
                else:
                    sigma *= 0.85

            # Fine refinement: smaller steps for precision
            sigma = max(sigma, 0.02)
            fine_calls = 0
            max_fine = min(40, budget - calls_used)
            for _ in range(max_fine):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                fine_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.03
                    update_best(x, fx)
                else:
                    sigma *= 0.90

        if best_x is None:
            # Fallback: should not happen, but handle gracefully
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
