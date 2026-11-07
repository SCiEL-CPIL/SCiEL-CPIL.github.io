"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with tuned sigma dynamics for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 300s wall-clock

        calls = 0
        best_f = np.inf
        best_x = None

        def tracked_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= self.budget:
                raise TimeoutError()
            calls += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        dim = self.dim

        # 5 diverse initializations
        num_starts = 5
        budget_per_start = self.budget // num_starts

        for start_idx in range(num_starts):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break

            # Different initial points for diversity
            if start_idx == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = tracked_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma with tuned dynamics: growth 1.25, decay 0.85
            sigma = 1.0 + 0.2 * start_idx
            success_count = 0
            total_count = 0
            stall_counter = 0

            while calls < self.budget and time.time() < deadline:
                # Adapt sigma based on success rate
                if total_count > 5:
                    success_rate = success_count / total_count
                    if success_rate > 0.3:
                        sigma *= 1.25  # Aggressive growth
                    elif success_rate < 0.1:
                        sigma *= 0.85  # Moderate decay
                    if total_count > 15:
                        success_count = 0
                        total_count = 0

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = tracked_func(y)
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    stall_counter = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    stall_counter += 1
                    # If stalled, try a larger jump
                    if stall_counter > 10:
                        y2 = np.clip(x + sigma * 3 * rng.standard_normal(dim), -5, 5)
                        fy2 = tracked_func(y2)
                        total_count += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma *= 0.9
                            stall_counter = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            # Restart from a random point
                            x = rng.uniform(-4, 4, dim)
                            fx = tracked_func(x)
                            sigma = 1.0
                            stall_counter = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
