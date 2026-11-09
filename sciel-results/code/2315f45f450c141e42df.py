"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for the BBOB f17 (Schaffers F7) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es_run(rng, x0, sigma0, budget_left):
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            m = max(4, int(round(4 + 3 * np.log(n))))  # population size
            # For n=5: m = max(4, int(round(4 + 3*1.609))) = max(4, int(round(8.827))) = max(4, 9) = 9
            # Let's use a fixed reasonable population size for 5D
            m = 10  # Fixed population size of 10 as suggested

            # CMA-ES parameters
            mu = m // 2  # number of selected individuals
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # Not used directly, but we need to track
            # For simplicity and speed in 5D, we'll use a simplified CMA-ES
            # Actually, let's implement a proper but efficient CMA-ES
            
            # Precompute Cholesky factor of C
            C_cholesky = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))

            fx = func(x)
            track_call(x, fx)
            
            # Evaluate initial population
            pop_x = np.zeros((m, n))
            pop_f = np.zeros(m)
            
            for i in range(m):
                if time.time() > deadline or calls_used >= total_budget:
                    return
                # Generate individual
                # Use C_cholesky to generate correlated steps
                z = rng.standard_normal(n)
                # Update C_cholesky approximation (simplified: just use identity for now in this simplified version)
                # For a proper CMA-ES, we need to update C_cholesky each generation
                # This is a simplified version that may not be as effective
                
                step = sigma * z  # Simplified: isotropic
                y = np.clip(x + step, -5, 5)
                pop_x[i] = y
                pop_f[i] = func(y)
                track_call(y, pop_f[i])

            # Sort by fitness
            order = np.argsort(pop_f)
            
            for _ in range(budget_left - m):  # budget_left includes the initial m evaluations
                if time.time() > deadline or calls_used >= total_budget:
                    return
                
                # Update mean
                x_new = np.sum(weights[:, None] * pop_x[order[:mu]], axis=0)
                
                # Update covariance matrix (simplified: skip full CMA update for speed)
                # In a full CMA-ES, we'd update C, pc, ps, sigma here
                # For this simplified version, we'll just update the mean and keep sigma fixed
                # or adapt it simply
                
                # Adapt sigma using simple success rate
                successes = sum(1 for i in range(m) if pop_f[order[i]] < fx)
                if successes > 0.3 * m:
                    sigma *= 1.1
                elif successes < 0.1 * m:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Move to new mean
                x = x_new
                fx = pop_f[order[0]]
                track_call(x, fx)  # Evaluate the new mean
                
                # Generate new population
                for i in range(m):
                    if time.time() > deadline or calls_used >= total_budget:
                        return
                    z = rng.standard_normal(n)
                    step = sigma * z
                    y = np.clip(x + step, -5, 5)
                    pop_x[i] = y
                    pop_f[i] = func(y)
                    track_call(y, pop_f[i])
                    
                # Sort by fitness
                order = np.argsort(pop_f)

        # Strategy:
        # 1. Brief random exploration to find a starting point
        # 2. Run CMA-ES from the best found point
        # 3. If time remains, run another CMA-ES from a different random point

        rng = np.random.default_rng(42)

        # Phase 1: Brief exploration
        explore_budget = max(5, min(20, total_budget // 25))
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            if calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        remaining = total_budget - calls_used

        # Phase 2: Main CMA-ES run from best point
        if remaining > 20:
            cma_es_run(rng, best_x, 1.0, remaining)

        remaining = total_budget - calls_used

        # Phase 3: Second CMA-ES run from a different random point if time allows
        if remaining > 20 and time.time() < deadline - 30:
            x2 = rng.uniform(-5, 5, dim)
            cma_es_run(rng, x2, 2.0, remaining)

        # Ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
