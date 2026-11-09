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
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)

        # Phase 1: Latin Hypercube-style sampling for fast early progress
        # Use a subset of the budget for LHS to get good initial coverage
        lhs_budget = min(50, budget // 10)
        x_best = None
        fx_best = np.inf

        # Generate LHS points
        lhs_points = np.zeros((lhs_budget, dim))
        for i in range(dim):
            perm = rng.permutation(lhs_budget)
            lhs_points[:, i] = (perm + 0.5) / lhs_budget * 10.0 - 5.0

        # Evaluate LHS points
        for i in range(lhs_budget):
            x = lhs_points[i]
            fx = func(x)
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        # Check deadline
        if time.time() > deadline:
            return x_best, fx_best

        # Phase 2: (1+1)-ES with periodic restarts to escape local minima
        # Katsuura has many local minima, so we use restarts
        remaining_budget = budget - lhs_budget
        sigma = 1.0
        x = x_best.copy()
        fx = fx_best

        # Track best for restart decisions
        fx_global_best = fx
        x_global_best = x.copy()
        no_improve_count = 0
        restart_threshold = 80  # restart after this many non-improving steps
        max_sigma = 5.0
        min_sigma = 0.01

        calls_made = lhs_budget

        for step in range(remaining_budget):
            # Deadline check every 10 steps for efficiency
            if step % 10 == 0 and time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, bounds, upper)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, max_sigma)
                no_improve_count = 0

                # Update global best
                if fx < fx_global_best:
                    fx_global_best = fx
                    x_global_best = x.copy()
            else:
                sigma = max(sigma * 0.9, min_sigma)
                no_improve_count += 1

            # Periodic restart: if stuck for too long, restart from a new random point
            # or from the global best with a larger step
            if no_improve_count >= restart_threshold:
                no_improve_count = 0
                # Restart from global best with larger sigma to escape basin
                x = x_global_best.copy()
                fx = fx_global_best
                sigma = max_sigma * 0.5

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
