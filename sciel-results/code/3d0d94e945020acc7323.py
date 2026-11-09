"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Use multiple restarts with different initial points
        # Each restart gets a portion of the budget
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize with different strategies per restart
            if restart == 0:
                x = np.zeros(self.dim)
            elif restart == 1:
                x = np.ones(self.dim) * 3.0
            else:
                rng_init = np.random.default_rng(42 + restart)
                x = rng_init.uniform(-4, 4, self.dim)

            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used = 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Adaptive step size with 1/5 rule
            successes = 0
            total_steps = 0
            no_improve = 0

            for _ in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * np.random.default_rng().standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                total_steps += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1

                # Adapt sigma every 10 steps using 1/5 success rule
                if total_steps % 10 == 0:
                    success_rate = successes / total_steps
                    if success_rate > 0.3:
                        sigma *= 1.3
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    successes = 0
                    total_steps = 0

                # Early stop if no improvement for a while and sigma is small
                if no_improve > 50 and sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
