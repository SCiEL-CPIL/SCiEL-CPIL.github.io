"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Multi-start strategy: split budget across several restarts
        # More restarts for multimodal functions, fewer for unimodal
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        last_restart_budget = total_budget - budget_per_restart * (n_restarts - 1)

        for r in range(n_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart at origin, others random
            if r == 0:
                x = np.zeros(n)
            else:
                # Sample from a wider region, biased towards center
                x = rng.uniform(-5, 5, n)
                # With some probability, sample near the best point found
                if best_x is not None and rng.random() < 0.3:
                    x = best_x + rng.normal(0, 1.0, n)
                    x = np.clip(x, -5, 5)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Adaptive step size with 1/5 success rule
            successes = 0
            window = 15
            window_count = 0

            remaining = min(budget_per_restart if r < n_restarts - 1 else last_restart_budget,
                           total_budget - calls_made)

            for _ in range(remaining):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    successes += 1
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma
                    sigma *= 1.25
                else:
                    # Decrease sigma
                    sigma *= 0.9

                window_count += 1
                if window_count >= window:
                    # 1/5 success rule
                    if successes / window_count > 0.2:
                        sigma *= 1.3
                    elif successes / window_count < 0.15:
                        sigma *= 0.7
                    successes = 0
                    window_count = 0

                # Clamp sigma
                sigma = max(0.01, min(sigma, 10.0))

            # End of restart

        return best_x, best_f
# EVOLVE-BLOCK-END
