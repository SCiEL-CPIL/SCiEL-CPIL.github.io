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

        def time_exceeded():
            return time.time() > deadline

        # Phase 1: Coarse exploration with (1+1)-ES
        sigma = 2.0
        x = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        evals = 1

        # 1/5 success rule parameters
        window = 10
        c_inc = 0.5
        successes = 0
        trial_count = 0

        while evals < budget and not time_exceeded():
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

            if f_new < best_f:
                successes += 1
            trial_count += 1

            if trial_count >= window:
                success_rate = successes / trial_count
                sigma *= np.exp(c_inc * (success_rate - 0.2))
                sigma = np.clip(sigma, 1e-6, 20.0)
                successes = 0
                trial_count = 0

            # Switch to fine-tuning phase when sigma is small
            if sigma < 0.1:
                break

        # Phase 2: Coordinate descent refinement
        if evals < budget and not time_exceeded():
            # Use remaining budget for coordinate descent
            remaining = budget - evals
            steps_per_coord = max(1, remaining // dim)
            
            for _ in range(steps_per_coord):
                for d in range(dim):
                    if evals >= budget or time_exceeded():
                        break
                    
                    # Try small perturbations in each coordinate
                    for delta in [0.01, -0.01, 0.001, -0.001]:
                        if evals >= budget or time_exceeded():
                            break
                        
                        x_try = best_x.copy()
                        x_try[d] += delta
                        np.clip(x_try, -5, 5, out=x_try)
                        
                        f_try = func(x_try)
                        evals += 1
                        
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
