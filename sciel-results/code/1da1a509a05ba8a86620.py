"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and restarts for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            val = func(x)
            calls_made += 1
            return float(val)

        # Phase 1: CMA-ES with adaptive population
        def cma_es(start_x, budget_for_cma, rng_local):
            nonlocal best_x, best_f, calls_made

            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Initial step size
            sigma = 1.0

            # Initial population
            pop = [start_x + sigma * rng_local.standard_normal(n) for _ in range(lam)]
            pop = [np.clip(p, -5.0, 5.0) for p in pop]
            fit = [wrapped_func(p) for p in pop]

            # Sort by fitness
            idx = np.argsort(fit)
            pop = [pop[i] for i in idx]
            fit = [fit[i] for i in idx]

            # CMA-ES state
            mean = np.mean([p[:mu] for p in pop], axis=0) * 0 + start_x  # start from initial
            mean = np.array(start_x, dtype=float)
            C = np.eye(n)
            cs = 1.0
            hs = 1.0
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_local_x = pop[0]
            best_local_f = fit[0]

            while calls_made < self.budget and (time.time() - start_time) < deadline:
                if calls_made >= budget_for_cma:
                    break

                # Generate new population
                new_pop = []
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    # Rank-mu update approximation (simplified)
                    new_pop.append(mean + sigma * z)
                new_pop = [np.clip(p, -5.0, 5.0) for p in new_pop]
                new_fit = [wrapped_func(p) for p in new_pop]

                idx = np.argsort(new_fit)
                new_pop = [new_pop[i] for i in idx]
                new_fit = [new_fit[i] for i in idx]

                # Update best
                if new_fit[0] < best_local_f:
                    best_local_f = new_fit[0]
                    best_local_x = new_pop[0]
                if best_local_f < best_f:
                    best_f = best_local_f
                    best_x = np.array(best_local_x)

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * new_pop[i]
                mean = mean + y_w

                # Update step size (CSA)
                d = np.sqrt((1 + 2 * max(0, cc / cs - 1)) * chi_n)
                hsig = (np.linalg.norm(y_w) / (sigma * chi_n)) ** (1 / (lam * 0.3)) < (1.4 + 2 / (n + 1)) * d
                cs = (1 - cs) + cs * (1 if hsig else 0)
                sigma *= np.exp(cs / (5 * (n + 2)) * (np.linalg.norm(y_w) / chi_n - 1))

                # Update covariance matrix
                delta_h = (1 - hsig) * cs * (1 - cc)
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * (np.outer(y_w + delta_h * mean, y_w + delta_h * mean) + delta_h * (1 - delta_h) * C)
                for i in range(mu):
                    v = (new_pop[i] - (mean - y_w)) / sigma
                    C += cmu * weights[i] * np.outer(v, v)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2

                # Stop if sigma is too small or too large
                if sigma < 1e-10 or sigma > 1e10:
                    break

            return best_local_x, best_local_f

        # Try multiple restarts with different initial points
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if calls_made >= self.budget or (time.time() - start_time) >= deadline:
                break

            # Different initialization strategies
            if r == 0:
                # Start at origin
                start_x = np.zeros(self.dim)
            elif r == 1:
                # Random point
                start_x = rng.uniform(-3, 3, self.dim)
            else:
                # Best found so far
                start_x = best_x if best_x is not None else rng.uniform(-3, 3, self.dim)

            budget_for_this = min(budget_per_restart, self.budget - calls_made)
            cma_es(start_x, budget_for_this, rng)

        # Phase 2: Fine-tuning with smaller steps around the best point
        if best_x is not None and calls_made < self.budget and (time.time() - start_time) < deadline:
            remaining = self.budget - calls_made
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if calls_made >= self.budget or (time.time() - start_time) >= deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
