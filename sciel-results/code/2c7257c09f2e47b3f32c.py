import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time management: 330s kill, use 320s as hard deadline
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Multi-start local search with 30 restarts x 10 evals
        # This provides broad coverage to find good basins for multimodal functions
        n_restarts = 30
        evals_per_restart = 10

        for r in range(n_restarts):
            if eval_count >= budget or time.time() > deadline - 5.0:
                break

            # Choose starting point: mix of random and perturbed best
            if r == 0:
                x = np.zeros(n)
            elif best_x is not None and rng.random() < 0.4:
                # Perturb the current best
                x = np.clip(best_x + rng.standard_normal(n) * 1.5, -5, 5)
            else:
                # Random start
                x = rng.uniform(-5, 5, n)

            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Fixed-step hill climb for this restart
            step_size = 1.0
            for _ in range(evals_per_restart - 1):
                if eval_count >= budget or time.time() > deadline - 5.0:
                    break

                # Try a random direction with fixed step
                z = rng.standard_normal(n)
                y = np.clip(x + step_size * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Shrink step for next try in this restart
                    step_size *= 0.7

        # Phase 2: (1+1)-ES refinement from the best point found
        if eval_count < budget and time.time() < deadline - 5.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            sigma_min = 1e-8
            sigma_max = 10.0

            while eval_count < budget and time.time() < deadline - 3.0:
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.3, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                if sigma < sigma_min:
                    break

        return best_x, best_f
