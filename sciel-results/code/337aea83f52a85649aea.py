"""Optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # We'll use a multi-start strategy with CMA-ES-like behavior
        # For f21 (Gallagher 101 peaks), multi-start is crucial due to many local minima

        rng = np.random.default_rng(42)

        # Track best overall
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        # Multi-start: use first ~20% of budget for diverse starts
        n_starts = max(5, budget // 10)  # at least 5, at most 10% of budget
        remaining_budget = budget - n_starts

        # Phase 1: Multi-start random sampling
        start_points = []
        for i in range(n_starts):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            start_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if not start_points:
            return best_x, best_f

        # Sort starts by fitness
        start_points.sort(key=lambda t: t[1])

        # Phase 2: Run local optimization from top starts
        # Use a (1+1)-ES with adaptive step size, similar to seed but smarter
        # Allocate remaining budget among top starts

        # Take top K starts for local optimization
        K = min(len(start_points), max(3, remaining_budget // 30))
        top_starts = start_points[:K]

        # Budget per local run
        budget_per_run = max(10, remaining_budget // K)

        for xi, fxi in top_starts:
            if time.time() > deadline:
                break

            x = xi.copy()
            fx = fxi
            sigma = 1.0  # initial step size
            dim_local = dim

            # Adaptive (1+1)-ES with 1/5 success rule
            successes = 0
            total = 0
            steps = 0
            max_steps = budget_per_run

            while steps < max_steps:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim_local), -5, 5)
                fy = func(y)
                steps += 1
                total += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.3
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85

                # 1/5 success rule: adjust sigma every 10 steps
                if total % 10 == 0:
                    rate = successes / 10.0
                    if rate > 0.5:
                        sigma *= 1.5
                    elif rate < 0.1:
                        sigma *= 0.5
                    successes = 0

                # If sigma becomes too small, try a larger restart
                if sigma < 0.01:
                    # Try a few random restarts from current best
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        y = np.clip(x + rng.uniform(-1, 1, dim_local), -5, 5)
                        fy = func(y)
                        steps += 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma = 0.5
                            break
                    else:
                        break

                # Also try if sigma gets too large
                if sigma > 10.0:
                    sigma = 10.0

        return best_x, best_f
# EVOLVE-BLOCK-END
