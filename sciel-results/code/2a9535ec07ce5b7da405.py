"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f14 (Different powers):
    1. Latin Hypercube Sampling for global coverage (first ~60 calls)
    2. CMA-ES (via scipy) for local refinement from the best LHS point
    3. Budget guard to stay within 500 calls and time limit
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 280s safety margin

        calls = [0]

        def eval_func(x):
            if calls[0] >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            return func(x)

        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = min(50, self.budget // 10)
        rng = np.random.default_rng(42)
        lhs_points = self._latin_hypercube(n_lhs, self.dim, rng)

        best_x = None
        best_f = float('inf')

        for x in lhs_points:
            fx = eval_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: single random point
            x = rng.uniform(-5, 5, self.dim)
            fx = eval_func(x)
            best_x = x
            best_f = fx

        # Phase 2: Local refinement using Nelder-Mead or Powell from best point
        # Use scipy.optimize.minimize with a tight budget
        remaining = self.budget - calls[0]
        if remaining > 10:
            try:
                from scipy.optimize import minimize
                # Use Powell's method which doesn't need gradients
                # Limit function evaluations
                result = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': min(remaining, 200)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If still have budget, do a few more local steps
        remaining = self.budget - calls[0]
        if remaining > 5:
            sigma = 0.5
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f

    def _latin_hypercube(self, n, dim, rng):
        """Generate n points using Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Random permutation for each dimension
            perm = rng.permutation(n)
            # Uniform within each stratum
            points[:, d] = (perm + rng.uniform(size=n)) / n * 10 - 5
        return points
# EVOLVE-BLOCK-END
