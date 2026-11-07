"""Rewritten program using multi-start CMA-ES with adaptive step sizing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using multi-start local search with adaptive step sizing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        calls = 0
        x_best = None
        f_best = float('inf')

        def update_best(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        def local_search(x_start, f_start, max_calls, sigma_init=1.0):
            nonlocal calls, f_best, x_best
            x = x_start.copy()
            fx = f_start
            sigma = sigma_init
            n_success = 0
            n_total = 0
            last_improvement = 0
            calls_in_search = 0

            while calls_in_search < max_calls and time.time() < deadline:
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)

                fy = func(y)
                calls += 1
                calls_in_search += 1

                if fy < fx:
                    x = y
                    fx = fy
                    n_success += 1
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                        last_improvement = calls_in_search
                    sigma *= 1.05
                else:
                    n_total += 1
                    sigma *= 0.95

                # Adaptive sigma adjustment based on success rate
                if n_total > 20:
                    success_rate = n_success / (n_success + n_total)
                    # 1/5 rule: adjust sigma based on success rate
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    n_success = 0
                    n_total = 0

                if sigma < 1e-10:
                    break

                # Check for stagnation
                if calls_in_search - last_improvement > 50:
                    # Try a larger jump
                    y = x + 2.0 * sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    calls_in_search += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < f_best:
                            f_best = fx
                            x_best = x.copy()
                        last_improvement = calls_in_search

            return x, fx

        # Phase 1: Quick exploration with multiple random starts
        n_initial = min(10, self.budget // 50)
        for i in range(n_initial):
            if time.time() > deadline or calls >= self.budget:
                break
            x_try = rng.uniform(-5, 5, self.dim)
            f_try = func(x_try)
            calls += 1
            update_best(x_try, f_try)

        # Phase 2: Local search from best point
        if x_best is not None and time.time() < deadline and calls < self.budget:
            remaining = self.budget - calls
            # Use 70% of remaining budget for first local search
            max_calls = int(remaining * 0.7)
            x, fx = local_search(x_best, f_best, max_calls, sigma_init=1.5)

            # Phase 3: If budget remains, try a few more restarts from different points
            if time.time() < deadline and calls < self.budget:
                n_restarts = min(5, (self.budget - calls) // 30)
                for r in range(n_restarts):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    # Generate a point in the direction of improvement from best
                    if rng.random() < 0.5:
                        x_try = x_best + rng.uniform(-2, 2, self.dim)
                    else:
                        x_try = rng.uniform(-5, 5, self.dim)
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = func(x_try)
                    calls += 1

                    if f_try < f_best:
                        update_best(x_try, f_try)
                        # Local search from this new best
                        if time.time() < deadline and calls < self.budget:
                            max_calls = min(20, self.budget - calls)
                            x, fx = local_search(x_try, f_try, max_calls, sigma_init=1.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
