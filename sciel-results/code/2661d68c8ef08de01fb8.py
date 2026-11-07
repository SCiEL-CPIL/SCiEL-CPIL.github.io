"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, deadline guard, and restart strategy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = None
        best_f = np.inf

        # We'll use a restart strategy: run (1+1)-ES for a portion of budget,
        # then restart from the best point with a fresh sigma
        # Split budget into segments for restarts
        num_segments = 3
        segment_budget = budget // num_segments
        calls_used = 0

        for seg in range(num_segments):
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Initialize: for first segment, start at center; for subsequent, start near best
            if seg == 0:
                x = np.zeros(dim)
            else:
                # Start near best with small perturbation
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            calls_used += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0 if seg == 0 else 1.0

            # (1+1)-ES loop for this segment
            for _ in range(segment_budget - 1):
                # Check deadline
                if time.time() > deadline - 5.0:
                    return best_x, best_f

                # Check budget
                if calls_used >= budget:
                    return best_x, best_f

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    # Increase sigma on success (slower growth)
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease sigma on failure (slower decay)
                    sigma *= 0.92

                # Safety: keep sigma within reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
