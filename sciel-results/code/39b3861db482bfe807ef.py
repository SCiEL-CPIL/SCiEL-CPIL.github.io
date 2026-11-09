"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start, adaptive step-size, and polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time() > 5.0

        # Multi-start: try both known basin regions of Lunacek bi-Rastrigin
        # Basin 1: near origin, Basin 2: near the shifted point
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 2.5,
            np.ones(dim) * -2.5,
            rng.uniform(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf
        calls = 0

        # Phase 1: Multi-start (1+1)-ES with adaptive step-size
        for init in init_points:
            if not time_left() or calls >= budget:
                break

            x = init.copy()
            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Run (1+1)-ES from this start with adaptive sigma
            sigma = 1.5
            success_count = 0
            total_count = 0
            window = 10
            successes_in_window = 0

            for _ in range(budget - calls - 50):
                if not time_left() or calls >= budget - 50:
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
                    # Gentle increase on success
                    sigma *= 1.15
                else:
                    # Gentle decrease on failure
                    sigma *= 0.92

                # Apply 1/5 success rule every 'window' steps
                if total_count % window == 0:
                    if successes_in_window >= window / 5.0:
                        sigma = min(sigma * 1.2, 5.0)
                    else:
                        sigma = max(sigma * 0.8, 1e-6)
                    successes_in_window = 0

                # Periodic restart if stuck
                if total_count % 50 == 0 and success_count / max(total_count, 1) < 0.1:
                    sigma = max(sigma * 2.0, 0.5)

        # Phase 2: Final polish phase with very small steps
        if best_x is not None and time_left() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            sigma_min = 1e-8
            sigma_max = 1e-3

            for _ in range(budget - calls):
                if not time_left() or calls >= budget:
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

                # Reduce sigma further as we converge
                if calls > budget * 0.9:
                    sigma = max(sigma * 0.98, sigma_min)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
