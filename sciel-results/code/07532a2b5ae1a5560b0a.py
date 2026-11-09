"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-start, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        # Multi-start: try several random starting points
        # Schwefel has many local minima, so diversity helps
        n_starts = 5
        per_budget = budget // n_starts
        if per_budget < 20:
            per_budget = 20
            n_starts = max(1, budget // per_budget)

        best_x = None
        best_f = np.inf

        for start_i in range(n_starts):
            # Check time
            if time.time() > deadline:
                break

            # Different starting strategies
            if start_i == 0:
                x = rng.uniform(-5, 5, dim)
            elif start_i == 1:
                x = np.zeros(dim)
            elif start_i == 2:
                x = rng.uniform(-2, 2, dim)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # (1+1)-ES for this start
            calls_made = 1
            for _ in range(per_budget - 1):
                # Check time
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87

                # Stop if sigma is too small
                if sigma < 1e-8:
                    break

            # If we still have budget left and time, continue with global search
            # for the remaining starts

        # Final polish: use remaining budget for a fine-grained local search
        remaining = budget - (n_starts * per_budget)
        if remaining > 10 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
