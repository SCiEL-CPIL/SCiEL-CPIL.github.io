"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with CMA-like adaptation, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf

        # We'll do multiple restarts. Each restart uses a portion of the budget.
        # Total calls must be <= budget.
        # Strategy: 8 restarts, each with budget//8 calls (last one gets remainder).
        n_restarts = 8
        calls_per_restart = budget // n_restarts
        remaining = budget - calls_per_restart * (n_restarts - 1)

        for r in range(n_restarts):
            if r < n_restarts - 1:
                calls_this = calls_per_restart
            else:
                calls_this = remaining

            # Initialize: mix of random and structured starts
            if r == 0:
                x = np.zeros(dim)  # origin as deterministic first start
            elif r == 1:
                x = np.full(dim, 2.0)
            elif r == 2:
                x = np.full(dim, -2.0)
            elif r == 3:
                x = rng.uniform(-4, 4, dim)
            elif r == 4:
                x = rng.uniform(-3, 3, dim)
            elif r == 5:
                x = rng.uniform(-4, 4, dim)
            elif r == 6:
                x = np.zeros(dim) + 0.1 * rng.standard_normal(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            x = np.clip(x, -5, 5)
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5  # initial step size
            # 1/5 success rule adaptation
            success_window = 10
            successes = 0
            window_count = 0

            for i in range(calls_this):
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)

                window_count += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass

                # Adapt sigma every window_count steps
                if window_count >= success_window:
                    rate = successes / window_count
                    if rate > 0.2:
                        sigma *= 1.25
                    elif rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                    window_count = 0

                # Keep sigma in reasonable bounds
                sigma = max(1e-4, min(sigma, 10.0))

            # If we broke early due to deadline, stop all restarts
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
