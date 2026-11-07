"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES with adaptive step size and stagnation-based restarts for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        remaining_budget = budget

        # Parameters for multi-start
        max_restarts = 10
        stagnation_limit = 25  # Restart if no improvement for 25 steps
        min_budget_per_restart = 15  # Don't start a new restart if less than 15 calls left

        restart_count = 0

        while remaining_budget > min_budget_per_restart and restart_count < max_restarts:
            if time.time() > deadline:
                break

            # Initialize new restart
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            remaining_budget -= 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            no_improve_count = 0
            success_window = []
            window_size = 10

            # Run local (1+1)-ES for this restart
            # Cap single restart length to encourage more restarts if budget allows
            current_restart_budget = min(remaining_budget, 80)

            for _ in range(current_restart_budget):
                if time.time() > deadline:
                    break
                if remaining_budget <= 0:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve_count = 0
                    success_window.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    success_window.append(0)
                    no_improve_count += 1

                # Adapt sigma based on 1/5th success rule
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05  # Too many successes, increase step
                    elif success_rate < 0.1:
                        sigma *= 0.95  # Too few successes, decrease step
                    # Keep window manageable
                    if len(success_window) > window_size:
                        success_window.pop(0)

                sigma = max(0.01, min(sigma, 5.0))

                # Stagnation check
                if no_improve_count >= stagnation_limit:
                    break

            restart_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
