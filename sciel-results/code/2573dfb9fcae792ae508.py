"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Track success rate for 1/5 rule
        n_success = 0
        n_total = 0
        sigma_min = 1e-10
        sigma_max = 10.0

        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            n_total += 1
            if fy < fx:
                x, fx = y, fy
                n_success += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                pass

            # Update sigma using 1/5 success rule
            # Check every few iterations to reduce overhead
            if n_total >= 10:
                success_rate = n_success / n_total
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.87
                n_success = 0
                n_total = 0

            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            elif sigma < sigma_min:
                sigma = sigma_min

        return best_x, best_fx
# EVOLVE-BLOCK-END
