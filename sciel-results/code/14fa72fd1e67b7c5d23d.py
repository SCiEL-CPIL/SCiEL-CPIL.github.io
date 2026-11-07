"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts, sigma-adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        total_budget = self.budget

        # Use multiple restarts to avoid getting stuck
        n_restarts = 4
        budget_per_restart = max(1, total_budget // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break

            # Different starting points for each restart
            if r == 0:
                x = np.array([0.0] * self.dim)
            elif r == 1:
                x = np.array([2.0] * self.dim)
            elif r == 2:
                x = np.array([-2.0] * self.dim)
            else:
                x = np.array([1.0, -1.0, 2.0, -2.0, 0.5][:self.dim])

            fx = func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            if time.time() > deadline - 10.0:
                break

            # (1+1)-ES with restart-aware sigma adaptation
            sigma = 1.5
            no_improve = 0
            max_no_improve = 80 + 20 * r

            for i in range(budget_per_restart):
                if time.time() > deadline:
                    return best_x, best_f
                if no_improve > max_no_improve:
                    break

                # Adaptive step: shrink slowly, grow on success
                direction = np.random.default_rng(42 + r * 1000 + i).standard_normal(self.dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.92

                    # Occasional larger exploratory jump
                    if no_improve % 40 == 0:
                        jump = np.random.default_rng(99 + r * 77 + i).standard_normal(self.dim)
                        yj = np.clip(x + 3.0 * jump, -5, 5)
                        fyj = func(yj)
                        if fyj < fx:
                            x, fx = yj, fyj
                            sigma = 1.5
                            no_improve = 0
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()

            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
