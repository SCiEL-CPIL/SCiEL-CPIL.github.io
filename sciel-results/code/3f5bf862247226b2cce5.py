"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule, time guard, and polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time() > 5.0

        # Multi-start: Lunacek bi-Rastrigin has two basins. Seed both plus a
        # random point so we don't get stuck in the wrong one.
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 2.5,
            np.ones(dim) * -2.5,
            rng.uniform(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf
        calls = 0

        # Reserve some budget for the polish phase.
        polish_reserve = 60
        main_budget = max(1, budget - polish_reserve)

        # Phase 1: Multi-start (1+1)-ES with 1/5 success rule.
        for init in init_points:
            if not time_left() or calls >= main_budget:
                break

            x = init.copy()
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            sigma_min = 1e-6
            sigma_max = 5.0
            window = 10
            successes_in_window = 0
            total_count = 0
            success_count = 0

            while calls < main_budget:
                if not time_left():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    successes_in_window += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.05, sigma_max)
                else:
                    sigma = max(sigma * 0.95, sigma_min)

                # 1/5 success rule on a rolling window.
                if total_count % window == 0:
                    if successes_in_window >= window / 5.0:
                        sigma = min(sigma * 1.2, sigma_max)
                    else:
                        sigma = max(sigma * 0.8, sigma_min)
                    successes_in_window = 0

                # Stuck detection: boost sigma to escape a local plateau.
                if total_count % 50 == 0 and success_count / max(total_count, 1) < 0.1:
                    sigma = max(sigma * 2.0, 0.5)

        # Phase 2: Polish phase with small bounded steps.
        if best_x is not None and time_left() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            sigma_min = 1e-8
            sigma_max = 1e-3

            while calls < budget:
                if not time_left():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, sigma_max)
                else:
                    sigma = max(sigma * 0.95, sigma_min)

                # Squeeze harder near the end of the budget.
                if calls > budget * 0.9:
                    sigma = max(sigma * 0.98, sigma_min)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
