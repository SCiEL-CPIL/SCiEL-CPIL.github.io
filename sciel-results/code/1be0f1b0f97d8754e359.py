"""Improved (1+1)-ES with adaptive step, restarts, and time guard for Katsuura f23."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with periodic restarts and time guard for multimodal Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        calls = 0

        # Best solution found so far
        best_x = None
        best_f = np.inf

        # Strategy parameters
        sigma = 1.5
        # Success-rate based step adaptation
        successes = 0
        window = 10
        count_in_window = 0

        # Restart logic: if no improvement for many iterations, restart
        stagnation = 0
        max_stagnation = 60

        def evaluate(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = evaluate(x)
        best_x = x.copy()
        best_f = fx

        for it in range(self.budget - 1):
            # Time guard
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5, 5)
            fy = evaluate(y)

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Adaptation
            count_in_window += 1
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                stagnation = 0
            else:
                stagnation += 1

            if count_in_window >= window:
                success_rate = successes / count_in_window
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.15:
                    sigma *= 0.9
                else:
                    sigma *= 1.0
                sigma = np.clip(sigma, 0.01, 10.0)
                successes = 0
                count_in_window = 0

            # Restart if stuck
            if stagnation >= max_stagnation:
                # Restart from best found so far with fresh sigma
                x = best_x + 0.1 * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
                fx = evaluate(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.5
                stagnation = 0
                successes = 0
                count_in_window = 0

        return best_x, best_f
