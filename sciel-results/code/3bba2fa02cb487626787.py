# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Portfolio optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Use multiple random starting points with L-BFGS-B (fast gradient-free via finite diff).
    2. For each start, do a short local refinement.
    3. Track best result across all runs.
    4. Use remaining budget for a final (1+1)-ES polish on the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def _count_call(self):
        self._call_count += 1

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Wrap func to count calls
        def wrapped(x):
            self._count_call()
            return func(x)

        # Initial best
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf

        # Phase 1: Multiple L-BFGS-B runs from different starting points
        # Use a subset of budget for these
        n_starts = 5
        budget_per_start = max(10, (self.budget - 50) // n_starts)

        rng = np.random.default_rng(42)

        for i in range(n_starts):
            if self._call_count >= self.budget or time.time() > deadline:
                break

            # Different starting points
            if i == 0:
                x0 = np.zeros(self.dim)
            elif i == 1:
                x0 = np.full(self.dim, 2.0)
            elif i == 2:
                x0 = np.full(self.dim, -2.0)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': budget_per_start, 'maxfun': budget_per_start}
                )
                if res.fun < self._best_f:
                    self._best_f = res.fun
                    self._best_x = np.array(res.x)
            except Exception:
                pass

        # Phase 2: Polish with (1+1)-ES using remaining budget
        if self._best_x is not None and self._call_count < self.budget and time.time() < deadline:
            x = self._best_x.copy()
            fx = self._best_f
            sigma = 0.5

            while self._call_count < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < self._best_f:
                        self._best_f = fx
                        self._best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
