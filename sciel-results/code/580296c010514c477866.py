"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)

        # Multiple restarts with decreasing sigma
        max_restarts = 10
        remaining = budget - 1

        for restart in range(max_restarts):
            if remaining <= 0:
                break
            if time.time() > deadline - 5.0:
                break

            # For first restart, start at center; for others, perturb best
            if restart == 0:
                x = np.zeros(dim)
            else:
                # Perturb around the best found so far
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            remaining -= 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma for this restart
            sigma = 1.0 / (restart + 1)
            min_sigma = 1e-4
            max_sigma = 5.0

            # Run (1+1)-ES for this restart
            while remaining > 0:
                if time.time() > deadline - 5.0:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                if sigma < min_sigma:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
