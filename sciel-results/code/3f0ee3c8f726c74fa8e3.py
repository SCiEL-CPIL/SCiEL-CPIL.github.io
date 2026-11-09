"""Combined optimizer: LHS initialization + (1+1)-ES with stagnation restarts and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with LHS initialization, stagnation restarts, and Nelder-Mead polish for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n points in [-5,5]^dim using Latin Hypercube sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            perm = rng.permutation(n)
            points[:, d] = lower[perm] + rng.uniform(0, 1, n) * (upper[perm] - lower[perm])
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Latin Hypercube sampling for fast initial basin discovery
        n_lhs = min(40, budget // 10)
        lhs_points = self._latin_hypercube(n_lhs, dim, rng)
        lhs_fits = []

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            lhs_fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and stagnation restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        no_improve_count = 0
        stagnation_limit = 20
        restart_count = 0
        max_restarts = 5
        successes = 0
        window = 10
        trial_count = 0

        # Reserve last ~50 calls for Nelder-Mead polish
        nm_budget = 50
        es_budget = budget - calls_made - nm_budget

        for i in range(es_budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                successes += 1
                sigma *= 1.15
            else:
                sigma *= 0.85

            trial_count += 1
            if trial_count % window == 0:
                success_rate = successes / window
                if success_rate > 0.3:
                    sigma *= 1.3
                elif success_rate < 0.1:
                    sigma *= 0.7
                successes = 0

            sigma = np.clip(sigma, 0.01, 5.0)

            # Stagnation restart
            if fy < fx:
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= stagnation_limit and restart_count < max_restarts:
                restart_count += 1
                no_improve_count = 0
                # Perturb from global best
                x = best_x + sigma * 2.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = max(sigma, 0.5)

        # Phase 3: Nelder-Mead polish on the final best point
        if calls_made < budget and time.time() < deadline - 5:
            remaining = budget - calls_made
            nm_maxiter = min(remaining, 50)

            def nm_func(z):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return np.inf
                calls_made += 1
                return func(np.clip(z, -5, 5))

            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
