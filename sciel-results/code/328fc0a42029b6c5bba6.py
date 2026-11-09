"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-stage optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.
    Strategy:
      1. Deterministic coarse-to-fine grid search (Halton/LHS + local refinement)
         to identify the global basin early.
      2. CMA-ES (via scipy) for fine-tuning from the best point found.
      3. Time-budget guard to stay within 300s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = time.time() + 290.0  # 10s safety margin

    def _check_time(self):
        return time.time() < self._deadline

    def _halton_sequence(self, n, dim, base=None):
        """Generate n points in [0,1)^dim using Halton sequence."""
        if base is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            f = 1.0
            for i in range(1, n + 1):
                num = i
                while num > 0:
                    num, rem = divmod(num, b)
                    f /= b
                    points[i - 1, d] += rem * f
                f = 1.0
        return points

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._calls = 0

        def f(x):
            self._calls += 1
            return func(x)

        # Stage 1: Coarse global search using Halton sequence
        # Evaluate 200 points spread across [-5, 5]^5
        n_coarse = min(200, budget // 3)
        halton = self._halton_sequence(n_coarse, dim)
        # Map to [-5, 5]
        coarse_pts = halton * 10.0 - 5.0

        best_x = coarse_pts[0].copy()
        best_f = f(best_x)
        for i in range(1, n_coarse):
            if not self._check_time():
                break
            x = coarse_pts[i]
            fx = f(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Stage 2: Local refinement around the best point
        # Use a smaller neighborhood with more points
        if self._check_time() and self._calls < budget - 50:
            n_refine = min(100, budget - self._calls - 50)
            # Sample in a box of radius 2.0 around best_x
            radius = 2.0
            refine_halton = self._halton_sequence(n_refine, dim)
            refine_pts = best_x + (refine_halton * 2.0 - 1.0) * radius
            refine_pts = np.clip(refine_pts, -5.0, 5.0)

            for i in range(n_refine):
                if not self._check_time():
                    break
                x = refine_pts[i]
                fx = f(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Stage 3: CMA-ES fine-tuning from best point
        if self._check_time() and self._calls < budget - 10:
            from scipy.optimize import minimize

            # Use Nelder-Mead as a lightweight local optimizer
            # (CMA-ES via scipy is not directly available, so use NM)
            try:
                res = minimize(
                    f,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Stage 4: Final local search with small steps
        if self._check_time() and self._calls < budget:
            rng = np.random.default_rng(42)
            sigma = 0.5
            for _ in range(min(100, budget - self._calls)):
                if not self._check_time():
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
