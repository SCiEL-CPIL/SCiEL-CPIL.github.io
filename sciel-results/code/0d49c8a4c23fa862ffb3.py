"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES optimizer with 1/5 success rule for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # (1+1)-ES with 1/5 success rule
        # Initial step: cover a good portion of the box
        sigma = 1.5
        x = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        evals = 1

        # 1/5 success rule parameters
        # sigma *= exp(c_inc * (successes / last_n - 1/5))
        # Use a window of last N trials
        window = 15
        c_inc = 1.0 / 5.0  # standard 1/5 rule constant
        successes = 0
        trial_count = 0
        sigma_history = []

        def time_exceeded():
            return time.time() > deadline

        while evals < budget and not time_exceeded():
            # Generate candidate
            y = rng.standard_normal(dim)
            x_new = x + sigma * y
            np.clip(x_new, -5, 5, out=x_new)

            f_new = func(x_new)
            evals += 1

            improved = f_new < best_f
            if improved:
                best_f = f_new
                best_x = x_new.copy()
                x = x_new.copy()

            # Track success for 1/5 rule (success = improvement over current x)
            if f_new < best_f:
                successes += 1
            trial_count += 1

            # Adapt sigma using 1/5 rule on a sliding window
            if trial_count >= window:
                success_rate = successes / trial_count
                # 1/5 rule: if success rate > 1/5, increase sigma; else decrease
                sigma *= np.exp(c_inc * (success_rate - 0.2))
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 20.0)
                successes = 0
                trial_count = 0

            # If sigma is very small, we've likely converged
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
