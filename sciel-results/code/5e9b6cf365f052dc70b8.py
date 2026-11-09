"""Combined optimizer: LHS initialization + (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with LHS initialization and Nelder-Mead polish for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        points = np.zeros((n, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            perm = rng.permutation(n)
            points[:, d] = lower[perm] + (upper - lower)[perm] * rng.random(n)
        return points

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Latin Hypercube-style sampling for fast initial basin discovery
        n_lhs = min(50, budget // 10)
        lhs_points = self._latin_hypercube(n_lhs, dim, rng)
        for i in range(n_lhs):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        calls_made = n_lhs

        # Reserve some budget for Nelder-Mead polish
        nm_budget = 100
        es_limit = budget - nm_budget

        while calls_made < es_limit:
            # Deadline check with safety margin
            if time.time() - start_time > 315.0:
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
                sigma *= 1.1
            else:
                sigma *= 0.9

        # Phase 3: Nelder-Mead polish
        if calls_made < budget:
            nm_func_calls = 0
            def nm_func(x):
                nonlocal nm_func_calls
                if calls_made + nm_func_calls >= budget:
                    return np.inf
                nm_func_calls += 1
                return func(np.clip(x, -5, 5))
            
            # In 5D, Nelder-Mead uses ~2*dim function evaluations per iteration
            maxiter = max(1, (budget - calls_made) // (2 * dim))
            
            res = minimize(nm_func, best_x, method='Nelder-Mead', 
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
