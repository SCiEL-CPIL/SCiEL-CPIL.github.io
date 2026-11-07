"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer with multiple (1+1)-ES runs and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Multi-start (1+1)-ES with adaptive sigma
        n_starts = 8
        calls_per_start = self.budget // n_starts
        remaining_budget = self.budget

        for s in range(n_starts):
            if remaining_budget <= 0:
                break
            if time.time() > deadline - 10:
                break

            # Initialize starting point
            if s == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Mix of random and perturbation from current best
                if rng.random() < 0.3 and best_x is not None:
                    x = best_x + rng.standard_normal(self.dim) * 2.0
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            remaining_budget -= 1
            update_best(x, fx)

            sigma = 2.0
            no_improve = 0
            max_no_improve = 30

            while remaining_budget > 0 and no_improve < max_no_improve:
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.3
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

            # If we exhausted budget or hit deadline, break
            if remaining_budget <= 0 or time.time() > deadline - 5:
                break

        # Final local polish around the best point if we have budget left
        if remaining_budget > 10 and best_x is not None and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            for _ in range(min(remaining_budget, 50)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
