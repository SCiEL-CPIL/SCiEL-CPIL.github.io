"""Multi-restart (1+1)-ES for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    """Multi-restart (1+1)-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # Multi-restart (1+1)-ES
        # Strategy:
        # 1. Start at origin with moderate step size
        # 2. Use (1+1)-ES with step size adaptation
        # 3. Restart when no improvement for a while
        # 4. Keep global best across all restarts

        # Parameters for (1+1)-ES
        sigma_init = 1.0  # Initial step size
        sigma_min = 1e-6
        sigma_max = 5.0
        no_improve_limit = 20  # Restart after this many non-improving steps
        max_restarts = 5

        for restart in range(max_restarts):
            if time_exceeded() or evals >= budget:
                break

            # Initialize current point
            if restart == 0:
                x = np.zeros(dim)
                sigma = sigma_init
            else:
                # Restart near the best found so far, with some perturbation
                x = best_x + rng.normal(0, 1.0, size=dim)
                x = np.clip(x, -5, 5)
                sigma = max(sigma_min, sigma_init * 0.5)

            # Evaluate starting point
            try:
                f_x = func(x)
                evals += 1
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            except Exception:
                evals += 1
                f_x = np.inf

            no_improve = 0
            step = 0

            while not time_exceeded() and evals < budget:
                # Generate mutant
                y = x + sigma * rng.normal(size=dim)
                y = np.clip(y, -5, 5)

                try:
                    f_y = func(y)
                    evals += 1
                except Exception:
                    evals += 1
                    f_y = np.inf

                if f_y < f_x:
                    # Accept: move to y
                    x = y
                    f_x = f_y
                    no_improve = 0
                    # Increase sigma slightly on success
                    sigma = min(sigma_max, sigma * 1.05)

                    # Update global best
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                else:
                    # Reject: stay at x
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma_min, sigma * 0.95)

                step += 1

                # Check for stagnation
                if no_improve >= no_improve_limit:
                    break

                # If sigma is very small, break to restart
                if sigma <= sigma_min * 1.1:
                    break

        return best_x, best_f
