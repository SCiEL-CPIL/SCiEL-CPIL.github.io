# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Random Restart (1+1)-ES with adaptive step size for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # (1+1)-ES parameters
        # Start at origin
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # Initial step size, covers a good portion of [-5, 5]

        # Stagnation detection
        stagnation_counter = 0
        stagnation_threshold = 25  # Restart if no improvement for 25 steps

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Update current solution and sigma
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5  # Increase step size on success
                stagnation_counter = 0
            else:
                sigma *= 0.85  # Decrease step size on failure
                stagnation_counter += 1

            # Check for stagnation and restart
            if stagnation_counter >= stagnation_threshold:
                # Restart from a new random point
                # This helps escape local optima in multimodal functions
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                sigma = 2.0
                stagnation_counter = 0

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
