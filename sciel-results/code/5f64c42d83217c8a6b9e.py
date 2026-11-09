"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # Initial multi-start to find a good starting point
        n_starts = min(20, budget // 10)
        for _ in range(n_starts):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Main optimization loop with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        no_improve = 0
        max_no_improve = 30

        # Adaptive parameters
        sigma_min = 0.01
        sigma_max = 5.0
        sigma_inc = 1.3
        sigma_dec = 0.7

        while calls_made < budget and time_ok():
            # Generate candidate
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = evaluate(y)

            if fy < fx:
                # Accept
                x = y
                fx = fy
                no_improve = 0
                sigma = min(sigma * sigma_inc, sigma_max)

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Reject
                no_improve += 1
                sigma = max(sigma * sigma_dec, sigma_min)

            # Restart if stuck
            if no_improve >= max_no_improve:
                no_improve = 0
                # Try a new random start or perturb current best
                if rng.random() < 0.3:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = best_x + rng.standard_normal(dim) * 1.0
                    x = np.clip(x, -5, 5)
                fx = evaluate(x)
                sigma = 1.5

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
