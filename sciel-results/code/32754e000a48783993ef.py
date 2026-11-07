"""Optimizer for BBOB-style f6 (Attractive sector)."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        calls = 1
        self.budget

        # Multi-start with restarts
        # Strategy: run (1+1)-ES, if no improvement for a while, restart from a new random point
        # or from the current best with perturbation

        # Use a simpler approach: split budget into segments, restart if stagnating
        # But first, let's do a robust (1+1)-ES with adaptive sigma

        sigma = 2.0
        no_improve_count = 0
        stagnation_threshold = 30  # restart after 30 consecutive non-improvements

        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                no_improve_count = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.87
                no_improve_count += 1

            # Restart if stagnating
            if no_improve_count >= stagnation_threshold:
                # Restart from best found so far with a large perturbation
                x = np.clip(best_x + 1.0 * rng.standard_normal(self.dim), -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma = 2.0
                no_improve_count = 0

            if calls >= self.budget:
                break

        return best_x, best_fx
