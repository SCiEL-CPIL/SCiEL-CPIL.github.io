"""Rewritten optimizer for BBOB f6 (Attractive sector) with improved (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f6 (Attractive sector) in 5D with budget 500.
    
    Strategy:
      - Start at origin (reasonable prior since minimizer is in [-4,4]^5).
      - Use (1+1)-ES with adaptive step size based on success rate.
      - Aggressive sigma adaptation to converge quickly.
      - Time guard to ensure we finish within the 330s limit.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at origin
        x = np.zeros(dim)
        fx = func(x)
        calls_used = 1

        # (1+1)-ES parameters
        sigma = 2.0  # initial step size
        n_success = 0
        n_total = 0
        window_size = 8  # smaller window for faster adaptation

        for i in range(budget - calls_used):
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                n_success += 1
            n_total += 1

            # Adapt sigma based on success rate over the window
            if n_total >= window_size:
                success_rate = n_success / n_total
                # Target success rate around 1/5 = 0.2
                if success_rate > 0.25:
                    sigma *= 1.15  # increase step size
                elif success_rate < 0.15:
                    sigma *= 0.85  # decrease step size
                else:
                    sigma *= 1.0   # keep same
                n_success = 0
                n_total = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.005, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
