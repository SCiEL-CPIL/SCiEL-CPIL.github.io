"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time guard and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 1.0
        calls = 1

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Use a simple adaptation scheme
        success_count = 0
        window_size = 10
        window_successes = 0
        window_total = 0

        while calls < self.budget:
            if time.time() > deadline - 2.0:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                window_successes += 1
            else:
                pass

            window_total += 1
            if window_total >= window_size:
                # Adapt sigma based on success rate
                success_rate = window_successes / window_total
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                else:
                    sigma *= 1.0

                window_successes = 0
                window_total = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 2: Fine-tuning with smaller steps
        sigma_fine = 0.05
        remaining = self.budget - calls
        for _ in range(min(remaining, 50)):
            if time.time() > deadline - 1.0:
                break
            y = np.clip(x + sigma_fine * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
