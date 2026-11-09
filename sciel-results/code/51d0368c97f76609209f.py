"""CMA-ES based global optimizer for BBOB-style Rosenbrock (f8)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter):
            nonlocal calls_used, deadline, best_x, best_f

            n = dim
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            # Weights
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Strategy parameters
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # State variables
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            invC = np.eye(n)

            # Precompute
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial population
            best_local = mean.copy()
            best_local_f = np.inf

            for it in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Generate population
                pop = []
                for i in range(lambda_):
                    z = np.random.randn(n)
                    x = mean + sigma * (B @ (D * z))
                    x = np.clip(x, -5, 5)
                    pop.append(x)

                # Evaluate population
                fvals = np.array([call_func(x) for x in pop])
                
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_local_f:
                    best_local_f = fvals[min_idx]
                    best_local = pop[min_idx].copy()
                    if fvals[min_idx] < best_f:
                        best_f = fvals[min_idx]
                        best_x = pop[min_idx].copy()

                # Sort
                order = np.argsort(fvals)
                pop_sorted = np.array([pop[i] for i in order])
                fvals_sorted = fvals[order]

                # Check convergence
                if np.std(fvals_sorted[:mu]) < 1e-14:
                    break

                # Update mean
                old_mean = mean.copy()
                mean = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)

                # Update step-size control (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (D * ((mean - old_mean) / sigma)))
                # Note: The standard CSA formula uses C^{-1/2}, but for simplicity and stability 
                # in this dimension, we approximate or use the simplified version. 
                # Actually, let's use the proper one:
                # ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(np.linalg.cholesky(C), (mean - old_mean) / sigma)
                # But we don't have easy access to C^{-1/2} without updating C first.
                # Let's stick to a simplified CSA or standard one.
                # Standard CMA-ES:
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (mean - old_mean) / sigma
                # We will approximate C^{-1/2} as I for now to keep it simple, or compute it.
                # Let's compute it properly.
                L = np.linalg.cholesky(C)
                # C^{-1/2} x = L^{-T} L^{-1} x
                y = np.linalg.solve(L, (mean - old_mean) / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(L.T, y)

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * chiN

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (mean - old_mean)

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                    + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                    + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_sorted[:mu] - old_mean, (pop_sorted[:mu] - old_mean).T), axis=0)

                # Ensure C is symmetric
                C = (C + C.T) / 2.0

                # Update eigen decomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))

                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check termination
                if sigma < 1e-12:
                    break

            return best_local, best_local_f

        # Strategy:
        # 1. Multi-start CMA-ES
        # 2. Use diverse initial points
        # 3. Polish with a local method if needed

        rng = np.random.default_rng(42)

        # Generate diverse starts
        starts = [np.zeros(dim), np.ones(dim)]
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Sort by distance to 0? Or just evaluate?
        # Let's evaluate a few to pick good starting points
        start_results = []
        for s in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            s = np.clip(s, -5, 5)
            f_s = call_func(s)
            start_results.append((s, f_s))
        
        start_results.sort(key=lambda x: x[1])
        
        # Run CMA-ES from the best start
        if start_results:
            best_start_x, best_start_f = start_results[0]
            
            # Estimate sigma based on initial spread or just use a reasonable value
            sigma0 = 2.0
            
            # Allocate budget
            # CMA-ES typically needs ~ (n^2 + 4n) * 200 * log(f0/f*) evaluations
            # For n=5, that's ~ 45 * 200 * log(...) ~ 9000 * log(...)
            # With 500 budget, we can run for ~ 500 / (4 + 3*log(5)) ~ 500/12 ~ 40 iterations
            # Each iteration uses lambda_ = 4 + 3*log(5) ~ 12 evaluations
            # So ~ 40 iterations is good.
            max_iter = 100 # Will be limited by budget
            
            x_res, f_res = cma_es(best_start_x, sigma0, max_iter)

            # If we have budget left, try a second start
            remaining_calls = budget - calls_used
            if remaining_calls > 50 and len(start_results) > 1:
                second_start_x, second_start_f = start_results[1]
                if second_start_f < best_f:
                    x_res2, f_res2 = cma_es(second_start_x, sigma0, 50)

        # Final polish with a few steps of Nelder-Mead or Coordinate Descent if budget allows
        remaining_calls = budget - calls_used
        if remaining_calls > 10 and best_x is not None:
            # Simple coordinate descent polish
            x = best_x.copy()
            f_x = call_func(x)
            step = 0.5
            for _ in range(10):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(dim):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    x_trial = x.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = call_func(x_trial)
                    if f_trial < f_x:
                        x = x_trial
                        f_x = f_trial
                        improved = True
                    else:
                        x_trial = x.copy()
                        x_trial[i] -= step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = call_func(x_trial)
                        if f_trial < f_x:
                            x = x_trial
                            f_x = f_trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                else:
                    step = min(step * 1.2, 2.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
