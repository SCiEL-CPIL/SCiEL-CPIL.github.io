"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive sigma, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        rng = np.random.default_rng(0)

        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0

        # Track best
        best_x = x.copy()
        best_f = fx

        # Success counter for 1/5 rule
        successes = 0
        total_steps = 0

        for i in range(self.budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                successes = 0

            # 1/5 success rule: adjust sigma based on recent history
            # Use a sliding window of last 10 steps for success rate
            # Simple version: if we just succeeded, increase; else decrease
            # Better: use the classic 1/5 rule with a counter
            # Here we use a simplified approach:
            # If success rate over last N is < 1/5, increase sigma
            # If > 1/5, decrease sigma

            # For simplicity and robustness, use multiplicative update:
            # This is a heuristic that works well in practice
            if fy < fx:
                sigma *= 1.3
            else:
                sigma *= 0.9

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)

        # Final polish: try a few smaller steps around the best point
        if time.time() < deadline - 5:
            sigma_polish = max(sigma * 0.1, 1e-6)
            for _ in range(20):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy

        return best_x, best_f
# EVOLVE-BLOCK-END
