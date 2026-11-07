"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # (1+1)-ES parameters
        sigma = 1.0
        success_count = 0
        success_window = 0
        window_size = 10
        restarts = 0
        max_restarts = 3
        no_improve_calls = 0
        max_no_improve = 50

        while calls_used < budget and time.time() < deadline:
            # Generate trial point
            step = sigma * rng.standard_normal(dim)
            trial_x = best_x + step
            trial_x = np.clip(trial_x, -5, 5)
            
            trial_f = func(trial_x)
            calls_used += 1

            if trial_f < best_f:
                # Accept
                best_x = trial_x
                best_f = trial_f
                success_count += 1
                success_window += 1
                no_improve_calls = 0
            else:
                # Reject
                success_window += 1
                no_improve_calls += 1

            # Adapt sigma based on 1/5 success rule over window
            if success_window >= window_size:
                rate = success_count / success_window
                if rate > 0.5:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                else:
                    sigma *= 1.1
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)
                success_count = 0
                success_window = 0

            # Restart if stuck
            if no_improve_calls >= max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset sigma and restart from a random point near current best
                sigma = 1.0
                no_improve_calls = 0
                # Perturb best_x
                best_x = best_x + 0.5 * rng.standard_normal(dim)
                best_x = np.clip(best_x, -5, 5)
                best_f = func(best_x)
                calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
