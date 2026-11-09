"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)
        
        # Start at origin, as minimizer is in [-4, 4]^5
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        sigma = 2.0

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate a perturbed candidate
            # Use a mix of exploration and exploitation
            # The 1/5 rule logic: if success rate is around 1/5, sigma is optimal
            # Simple heuristic: if we succeed, increase sigma; if fail, decrease.
            
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept step
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase sigma to explore more
                sigma *= 1.2
                # Cap sigma to avoid huge jumps
                if sigma > 10.0:
                    sigma = 10.0
            else:
                # Reject step
                # Decrease sigma to refine
                sigma *= 0.9
                # Floor sigma
                if sigma < 1e-6:
                    sigma = 1e-6

        return best_x, best_f
# EVOLVE-BLOCK-END
