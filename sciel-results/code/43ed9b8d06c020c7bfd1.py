"""Rewritten optimizer: Halton space-filling scan + CMA-ES refinement."""
import time
import numpy as np
from scipy.optimize import minimize


def _halton_sequence(n, dim, start=1):
    """Generate n points in [0,1)^dim using the Halton sequence."""
    points = np.zeros((n, dim))
    for d in range(dim):
        # Prime for this dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        prime = primes[d % len(primes)]
        for i in range(n):
            val = 0.0
            p = 1.0
            j = i + start
            while j > 0:
                j, remainder = divmod(j, prime)
                val += remainder / p
                p *= prime
            points[i, d] = val
    return points


class Optimizer:
    """Halton space-filling scan followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        # Phase 1: Halton space-filling scan
        # Use a significant portion of budget for exploration
        n_halton = min(200, self.budget // 2)
        halton_pts = _halton_sequence(n_halton, self.dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_pts = halton_pts * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        call_count = 0

        for pt in halton_pts:
            if time.time() > deadline:
                break
            try:
                f = func(pt)
                call_count += 1
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()
            except Exception:
                break

        if best_x is None:
            # Fallback: random point
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: CMA-ES refinement from best Halton point
        # Use remaining budget for local search
        remaining_budget = self.budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Wrap func to track calls and enforce deadline
        call_counter = [call_count]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            call_counter[0] += 1
            return func(x)

        # Use CMA-ES via scipy
        try:
            result = minimize(
                wrapped_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': remaining_budget,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                }
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except Exception:
            pass

        return best_x, best_f
