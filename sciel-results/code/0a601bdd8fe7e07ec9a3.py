"""Combined optimizer: LHS initialization + (1+1)-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube-style sampling for fast early progress
        # Use a fraction of budget for exploration
        n_lhs = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        # Generate LHS points
        for i in range(n_lhs):
            # Check deadline
            if time.time() > deadline:
                break

            # LHS: divide each dimension into n_lhs equal intervals, pick one random point per interval
            x = np.zeros(dim)
            for d in range(dim):
                # Interval for this dimension
                interval_start = i / n_lhs
                interval_end = (i + 1) / n_lhs
                x[d] = -5.0 + 10.0 * (interval_start + rng.random() * (interval_end - interval_start))
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with periodic restarts from best point
        # Use remaining budget for exploitation
        remaining_budget = budget - n_lhs
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Adaptive step size with success rate tracking
        successes = 0
        trials = 0
        success_rate = 0.33  # target

        for _ in range(remaining_budget):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate trial point
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            trials += 1
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                # Increase step size on success
                sigma *= 1.2
            else:
                # Decrease step size on failure
                sigma *= 0.9

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)

            # Periodic restart: every 50 iterations, if no improvement in last 20, restart from best
            if trials % 50 == 0:
                # Update success rate
                if trials >= 20:
                    current_sr = successes / trials
                    # If success rate is very low, restart from best with smaller sigma
                    if current_sr < 0.05:
                        x = best_x.copy()
                        fx = best_f
                        sigma = 0.5
                        successes = 0
                        trials = 0

        return x, fx
# EVOLVE-BLOCK-END
