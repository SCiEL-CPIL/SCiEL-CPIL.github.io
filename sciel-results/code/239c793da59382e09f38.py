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

        # Phase 1: Coarse search to find the minimum region
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_used = 1

        # Adaptive step size
        sigma = 2.0
        successes = 0
        total = 0
        success_threshold = 0.2

        # Phase 1: Coarse exploration
        while calls_used < self.budget and time.time() < deadline:
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

            if total >= 5:
                rate = successes / total
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                successes = 0
                total = 0

            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10:
                sigma = 10

            # If we've converged somewhat, switch to fine-tuning
            if sigma < 0.1 and calls_used > 50:
                break

        # Phase 2: Fine-tuning with smaller steps
        if calls_used < self.budget and time.time() < deadline:
            sigma = 0.05
            successes = 0
            total = 0
            x = best_x.copy()
            fx = best_fx

            while calls_used < self.budget and time.time() < deadline:
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

                if total >= 5:
                    rate = successes / total
                    if rate > success_threshold:
                        sigma *= 1.2
                    elif rate < success_threshold:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                if sigma < 1e-8:
                    sigma = 1e-8
                if sigma > 1.0:
                    sigma = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
