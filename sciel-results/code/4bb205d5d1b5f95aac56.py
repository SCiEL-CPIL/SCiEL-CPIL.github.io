"""Deterministic grid-based global search followed by local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1 (80% budget): deterministic Halton sequence sampling to map the landscape.
      Phase 2 (20% budget): Nelder-Mead local refinement from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, count, dim, base=None):
        """Generate a Halton sequence in [0, 1)^dim."""
        if base is None:
            # First 'dim' prime numbers
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]

        points = []
        for i in range(1, count + 1):
            point = []
            for d, p in enumerate(base):
                f = 1.0
                r = 0.0
                n = i
                while n > 0:
                    f /= p
                    r += f * (n % p)
                    n //= p
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Halton sequence sampling (80% of budget)
        n_halton = int(budget * 0.8)
        # Reserve at least 50 calls for local refinement
        n_halton = max(n_halton, 10)
        n_local = budget - n_halton

        # Generate Halton points in [0, 1)^dim
        halton_pts = self._halton_sequence(n_halton, dim)

        # Map to [-5, 5]^dim
        halton_pts = halton_pts * 10.0 - 5.0

        # Evaluate all Halton points
        best_x = None
        best_f = np.inf

        for i in range(n_halton):
            x = halton_pts[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using Nelder-Mead
        if n_local > 0 and best_x is not None:
            # Use a wrapper that counts function evaluations
            calls_made = [0]
            max_calls = n_local

            def wrapped_func(x):
                if calls_made[0] >= max_calls:
                    return np.inf
                calls_made[0] += 1
                return func(np.clip(x, -5, 5))

            # Nelder-Mead options
            options = {
                'maxiter': max_calls,
                'maxfev': max_calls,
                'xatol': 1e-8,
                'fatol': 1e-10,
            }

            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options=options
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
