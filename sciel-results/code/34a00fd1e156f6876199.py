"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive step size for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def _time_ok():
            return (time.time() - start_time) < deadline

        def _eval(x):
            nonlocal calls_made, best_x, best_f
            if not _time_ok():
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-restart strategy: try several starting points
        # Include origin and diverse random points
        num_restarts = 5
        remaining = budget

        for restart in range(num_restarts):
            if remaining <= 10:
                break
            if not _time_ok():
                break

            # Choose starting point
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = rng.uniform(-4, 4, dim)
            elif restart == 2:
                x0 = rng.uniform(-2, 2, dim)
            elif restart == 3:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Use best found so far with perturbation
                if best_x is not None:
                    x0 = np.clip(best_x + rng.standard_normal(dim) * 1.0, -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = _eval(x0)
            if f0 is None:
                break
            remaining -= 1

            if f0 is None:
                break

            # CMA-ES-like local search from x0
            # Use a simple evolution strategy with adaptive sigma
            x = x0.copy()
            fx = f0
            sigma = 2.0
            success_count = 0
            window = 15
            successes_in_window = 0
            window_count = 0

            local_budget = min(remaining // max(1, num_restarts - restart), 150)
            local_budget = max(local_budget, 20)

            for step in range(local_budget):
                if remaining <= 0:
                    break
                if not _time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = _eval(y)
                if fy is None:
                    break
                remaining -= 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    successes_in_window += 1
                else:
                    sigma *= 0.9

                window_count += 1
                if window_count >= window:
                    success_rate = successes_in_window / window_count
                    # Adapt sigma based on success rate
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes_in_window = 0
                    window_count = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

            # After local search, if we have budget left and found good point,
            # do a fine-tuning phase with smaller sigma
            if remaining > 5 and _time_ok():
                fine_budget = min(remaining, 30)
                fine_sigma = 0.3
                for step in range(fine_budget):
                    if remaining <= 0:
                        break
                    if not _time_ok():
                        break
                    y = np.clip(x + fine_sigma * rng.standard_normal(dim), -5, 5)
                    fy = _eval(y)
                    if fy is None:
                        break
                    remaining -= 1
                    if fy < fx:
                        x = y
                        fx = fy
                        fine_sigma *= 1.1
                    else:
                        fine_sigma *= 0.85
                    fine_sigma = np.clip(fine_sigma, 0.01, 2.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
