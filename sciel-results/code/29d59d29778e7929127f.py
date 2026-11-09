"""Optimizer for BBOB f14 (different powers) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, robust initialization, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
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

        calls_made = len(candidates)
        if calls_made >= budget:
            return best_x, best_f

        x = best_x
        fx = best_f
        sigma = 1.0

        # (1+1)-ES loop
        for _ in range(budget - calls_made):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.85

            # Prevent sigma from becoming too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10.0:
                sigma = 10.0

        # Final high-precision polish pass: coordinate descent with shrinking steps
        if time.time() < deadline - 5.0:
            polish_calls = 0
            max_polish = min(50, budget - calls_made - (budget - calls_made - 0))  # use remaining budget
            # Actually, let's just do a fixed number of polish steps if we have time
            polish_steps = 100
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            for step in step_sizes:
                if time.time() > deadline - 2.0:
                    break
                for d in range(dim):
                    if time.time() > deadline - 2.0:
                        break
                    # Try positive and negative directions
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        polish_calls += 1
                        if fy < fx:
                            x, fx = y, fy

        return x, fx
# EVOLVE-BLOCK-END
