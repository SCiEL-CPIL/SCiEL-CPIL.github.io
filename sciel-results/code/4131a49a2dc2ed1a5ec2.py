# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: multi-start (1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_check():
            return time.time() > deadline

        # Strategy: multiple restarts with different initial points
        # Use a mix of random starts and structured starts
        # Phase 1: broad exploration with several restarts
        # Phase 2: fine-tuning around best point

        num_restarts = 8
        budget_per_restart = self.budget // num_restarts  # ~62 per restart
        extra_budget = self.budget - budget_per_restart * num_restarts

        for r in range(num_restarts):
            if calls_made >= self.budget or time_check():
                break

            # Choose initial point
            if r == 0:
                x = np.zeros(self.dim)
            elif r == 1:
                x = rng.uniform(-3, 3, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            no_improve = 0
            max_no_improve = 30

            for _ in range(budget_per_restart):
                if calls_made >= self.budget or time_check():
                    break
                if no_improve > max_no_improve:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

        # Phase 2: fine-tune around best point with small sigma
        if best_x is not None and calls_made < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = self.budget - calls_made
            for _ in range(remaining):
                if time_check():
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
