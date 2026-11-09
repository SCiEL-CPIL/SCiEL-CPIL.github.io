"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, multiple restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # We'll do multiple restarts, each with a portion of the budget
        # For Rastrigin rotated, we need good global exploration
        # Strategy: several restarts with decreasing budget, plus a final polish

        # Phase 1: Global exploration with several short runs
        n_restarts = 10
        budget_per_restart = max(1, (self.budget // 3) // n_restarts)
        remaining = self.budget

        for restart in range(n_restarts):
            if remaining <= 0:
                break
            if time.time() > deadline:
                break

            # Diverse starting points
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            remaining -= 1

            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.5
            step_count = 0
            max_steps = budget_per_restart
            consecutive_fail = 0

            for i in range(max_steps):
                if remaining <= 0:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                step_count += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    consecutive_fail = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                    consecutive_fail += 1
                    # If stuck for too long, break early
                    if consecutive_fail > 30 and i > 20:
                        break

        # Phase 2: Local refinement from the best point found
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            consecutive_fail = 0

            for i in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    consecutive_fail = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                    consecutive_fail += 1
                    if consecutive_fail > 50:
                        break

        # Phase 3: If still have budget, do a few more focused restarts near best
        if remaining > 10 and best_x is not None:
            for _ in range(min(5, remaining // 20)):
                if remaining < 20 or time.time() > deadline:
                    break
                x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                fx = func(x)
                remaining -= 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

                sigma = 0.3
                for i in range(min(remaining, 20)):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
