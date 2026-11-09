"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with robust initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Robust initialization: try origin and a few random points
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        calls_used = len(candidates)
        remaining = budget - calls_used

        if remaining <= 0:
            return best_x, best_f

        # (1+1)-ES with adaptive step size
        x = best_x
        fx = best_f
        sigma = 1.0

        # Try a few initial sigma values to pick the best one
        # Actually, let's just use a reasonable starting sigma
        sigma = 0.5

        for i in range(remaining):
            # Time guard
            if time.time() > deadline:
                break

            # Generate mutant
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y
                fx = fy
                # Increase step size on success
                sigma *= 1.3
            else:
                # Decrease step size on failure
                sigma *= 0.85

            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
