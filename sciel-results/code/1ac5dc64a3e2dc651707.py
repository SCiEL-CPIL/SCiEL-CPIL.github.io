"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer:
    - Starts with a CMA-like / (1+1)-ES with adaptive step size.
    - Includes a deadline guard to finish within wall-clock limits.
    - Uses a simple restart mechanism if progress stalls.
    - Explores with increasing diversity initially, then exploits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial population of candidates to find a good starting point
        # Use a few random starts and pick the best
        n_starts = min(10, self.budget // 10)
        if n_starts < 1:
            n_starts = 1

        best_x = None
        best_f = np.inf

        for _ in range(n_starts):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: single evaluation
            x = np.zeros(dim)
            best_f = float(func(x))
            best_x = x.copy()

        x = best_x
        fx = best_f
        sigma = 1.0
        no_improve_count = 0
        max_no_improve = 50

        calls_made = n_starts

        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Adaptive step size: increase on success, decrease on failure
            # Use a more aggressive strategy for sharp ridge
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = float(func(y))
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.2
                no_improve_count = 0
            else:
                sigma *= 0.8
                no_improve_count += 1

            # Restart if stuck
            if no_improve_count >= max_no_improve:
                # Try a few new random starts
                for _ in range(min(3, self.budget - calls_made)):
                    if time.time() > deadline:
                        break
                    x_new = rng.uniform(-5, 5, dim)
                    fx_new = float(func(x_new))
                    calls_made += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve_count = 0
                        sigma = 1.0
                        break
                else:
                    # If no improvement from restarts, reduce sigma further and continue
                    sigma *= 0.5
                    no_improve_count = 0

            # If sigma becomes too small, try a larger exploration step
            if sigma < 1e-6:
                sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
