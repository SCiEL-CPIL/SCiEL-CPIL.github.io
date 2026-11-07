"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        rng = np.random.default_rng(0)
        
        # Initial point: start at origin (often better for BBOB)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Step size
        sigma = 2.0

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            # Clip to box
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Increase step size on success (aggressive exploration)
                sigma *= 1.1
            else:
                # Failure
                # Decrease step size on failure (exploitation)
                sigma *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
