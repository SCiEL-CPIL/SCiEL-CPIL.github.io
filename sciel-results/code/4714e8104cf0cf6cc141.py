"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES style optimizer with adaptive step size for the Bent Cigar function.
    Uses a deadline guard to stay within the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Initialize at origin (Bent Cigar has a known structure around 0 after shift)
        # But we don't know the shift, so start at 0 and let the optimizer find it
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_used = 1

        # Adaptive step size
        sigma = 1.0
        # Success counter for 1/5 rule
        successes = 0
        total = 0
        success_threshold = 0.2  # 1/5

        while calls_used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                successes += 1
            total += 1

            # Adapt step size based on success rate
            if total >= 5:
                rate = successes / total
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                # Reset counters
                successes = 0
                total = 0

            # If sigma becomes too small or too large, adjust
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10:
                sigma = 10

        return best_x, best_fx
# EVOLVE-BLOCK-END
