"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restart strategy for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Initial exploration with multiple restarts
        # Use a population-based approach initially, then refine with (1+1)-ES

        # Strategy: Run several short (1+1)-ES runs with different starting points
        # and keep the best. Then continue refining the best with remaining budget.

        num_initial_runs = 5
        budget_per_run = max(50, (total_budget - 100) // num_initial_runs)
        remaining_budget = total_budget - num_initial_runs * budget_per_run

        for run_idx in range(num_initial_runs):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 30:
                break

            # Different starting points for diversity
            if run_idx == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Mix of random and center-biased starts
                if run_idx % 2 == 0:
                    x = rng.uniform(-3, 3, dim)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Run (1+1)-ES for this budget
            run_calls = 0
            max_run_calls = budget_per_run

            while run_calls < max_run_calls and calls_made < total_budget:
                if time.time() > deadline - 10:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                run_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                    # If sigma gets too small, try a larger jump
                    if sigma < 0.01:
                        y2 = np.clip(x + rng.uniform(-3, 3, dim), -5, 5)
                        fy2 = func(y2)
                        calls_made += 1
                        run_calls += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma = 1.0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            sigma = 0.5

        # Phase 2: Refine the best solution with remaining budget
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            while calls_made < total_budget:
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    best_f = fx
                    best_x = x.copy()
                else:
                    sigma *= 0.9

                    # Adaptive restart if stuck
                    if sigma < 0.005:
                        y2 = np.clip(x + rng.uniform(-1, 1, dim), -5, 5)
                        fy2 = func(y2)
                        calls_made += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma = 0.3
                            best_f = fx
                            best_x = x.copy()
                        else:
                            sigma = 0.1

        return best_x, best_f
# EVOLVE-BLOCK-END
