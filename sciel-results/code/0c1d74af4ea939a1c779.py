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
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # (1+1)-ES parameters
        sigma = 1.0
        success_count = 0
        success_window = 0
        window_size = 15
        restarts = 0
        max_restarts = 5
        no_improve_calls = 0
        max_no_improve = 40
        sigma_min = 1e-8
        sigma_max = 10.0

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
                if rate > 0.6:
                    sigma *= 1.4
                elif rate < 0.15:
                    sigma /= 1.4
                else:
                    sigma *= 1.05
                sigma = max(sigma, sigma_min)
                sigma = min(sigma, sigma_max)
                success_count = 0
                success_window = 0

            # Restart if stuck
            if no_improve_calls >= max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset sigma and restart from a random point
                sigma = 1.0
                no_improve_calls = 0
                # Try a few random restarts to find a better starting point
                for _ in range(3):
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    candidate_x = rng.uniform(-4, 4, size=dim)
                    candidate_f = func(candidate_x)
                    calls_used += 1
                    if candidate_f < best_f:
                        best_x = candidate_x
                        best_f = candidate_f

        # Final polish: use remaining budget for a fine-grained local search
        # around the best point found, with very small sigma
        if calls_used < budget and time.time() < deadline:
            polish_sigma = 0.01
            polish_no_improve = 0
            max_polish_no_improve = 20
            
            while calls_used < budget and time.time() < deadline:
                step = polish_sigma * rng.standard_normal(dim)
                trial_x = best_x + step
                trial_x = np.clip(trial_x, -5, 5)
                
                trial_f = func(trial_x)
                calls_used += 1

                if trial_f < best_f:
                    best_x = trial_x
                    best_f = trial_f
                    polish_no_improve = 0
                else:
                    polish_no_improve += 1
                    polish_sigma *= 0.9
                    polish_sigma = max(polish_sigma, 1e-6)
                
                if polish_no_improve >= max_polish_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
