"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-step coordinate descent with random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls
            fx = func(x)
            calls += 1
            return fx

        def coordinate_descent(x0, max_calls, step_init=2.0):
            """Adaptive coordinate descent with multi-step attempts."""
            nonlocal best_x, best_f, calls, deadline
            x = x0.copy()
            fx = call_func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            step = step_init
            no_improve = 0
            max_no_improve = 30

            while calls < max_calls and (time.time() - start_time) < deadline:
                improved = False
                for i in range(self.dim):
                    if calls >= max_calls or (time.time() - start_time) >= deadline:
                        break

                    # Try both directions with adaptive step
                    for direction in (1, -1):
                        if calls >= max_calls or (time.time() - start_time) >= deadline:
                            break

                        # Multi-step: try several step sizes
                        for s in (step, step * 0.5, step * 0.25, step * 0.1):
                            if calls >= max_calls or (time.time() - start_time) >= deadline:
                                break

                            x_new = x.copy()
                            x_new[i] += direction * s
                            x_new = np.clip(x_new, -5, 5)

                            fx_new = call_func(x_new)
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f, best_x = fx, x.copy()
                                # Increase step for this coordinate
                                step = min(step * 1.3, 10.0)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    # No coordinate improved
                    no_improve += 1
                    step *= 0.7
                    if no_improve >= max_no_improve:
                        break
                    continue
                # Reset no_improve if any improvement
                no_improve = 0

            return x, fx

        # Phase 1: Random restarts to find good basins
        # Use ~40% of budget for exploration, 60% for exploitation
        explore_budget = int(self.budget * 0.4)
        exploit_budget = self.budget - explore_budget

        # Try several random starting points
        n_restarts = 8
        for r in range(n_restarts):
            if calls >= self.budget or (time.time() - start_time) >= deadline:
                break

            if r == 0:
                x_start = np.zeros(self.dim)
            else:
                x_start = rng.uniform(-4.5, 4.5, self.dim)

            # Quick local search from this start
            max_calls_this = min(explore_budget // max(1, n_restarts - r), self.budget - calls)
            if max_calls_this < 5:
                break

            _, fx = coordinate_descent(x_start, calls + max_calls_this, step_init=2.0)

        # Phase 2: Exploit the best point found with more budget
        if best_x is not None and calls < self.budget and (time.time() - start_time) < deadline:
            remaining = self.budget - calls
            if remaining > 10:
                coordinate_descent(best_x.copy(), self.budget, step_init=1.0)

        # Phase 3: Final fine-tuning with very small steps
        if best_x is not None and calls < self.budget and (time.time() - start_time) < deadline:
            remaining = self.budget - calls
            if remaining > 10:
                coordinate_descent(best_x.copy(), self.budget, step_init=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
