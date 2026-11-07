"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with proven 1.5/0.87 adaptation for BBOB f13 sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at box center to maximize early convergence
        x_best = np.zeros(dim)
        f_best = func(x_best)
        calls = 1

        # (1+1)-ES with proven 1.5/0.87 adaptation rates
        step_size = 2.0  # initial step size for 5D box [-5,5]

        while calls < budget:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Generate mutant
            y = x_best + step_size * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            # Adapt step size using proven 1.5/0.87 rule
            if fy < f_best:
                # Success: increase step size
                step_size *= 1.5
                f_best = fy
                x_best = y
            else:
                # Failure: decrease step size
                step_size *= 0.87

            # Clamp step size to reasonable range
            step_size = max(0.001, min(step_size, 10.0))

        return x_best, f_best
# EVOLVE-BLOCK-END
