"""Optimizer for BBOB f20 (Schwefel) with restarts and adaptive step size."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        # Global best
        best_x = None
        best_f = np.inf

        # We'll do multiple restarts. Each restart uses a portion of the budget.
        # Strategy: start with a larger initial population, then refine.
        # For Schwefel, the global minimum is at (4.209687...,)^dim
        # But the function is rotated and shifted, so we need to search.

        # Use a combination of:
        # 1. Random sampling to find a good region
        # 2. CMA-ES-like local search from the best point

        # Phase 1: Random sampling to find a good starting point
        n_sample = min(50, total_budget // 5)
        rng = np.random.default_rng(42)

        samples_x = []
        samples_f = []
        for i in range(n_sample):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            samples_x.append(x)
            samples_f.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Local search from the best point using adaptive step size
        # Use a (1+lambda)-ES with adaptive sigma
        remaining = total_budget - len(samples_f)
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Adaptive local search
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size
        lambda_ = 10  # number of offspring per generation

        # Track progress
        no_improve = 0
        max_no_improve = 50  # restart local search if no improvement for this many generations

        while remaining > 0 and time.time() < deadline:
            # Generate lambda offspring
            for _ in range(lambda_):
                if remaining <= 0 or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.2  # increase step size on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1

            # If no improvement for a while, reduce sigma and possibly restart
            if no_improve > max_no_improve:
                sigma *= 0.5
                if sigma < 0.01:
                    # Restart from a new random point
                    no_improve = 0
                    sigma = 2.0
                    if remaining > 10:
                        x = rng.uniform(-5, 5, dim)
                        fx = func(x)
                        remaining -= 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

            # Occasionally try a larger jump to escape local minima
            if no_improve > 20 and no_improve % 20 == 0:
                jump = x + 3.0 * rng.standard_normal(dim)
                jump = np.clip(jump, -5, 5)
                fj = func(jump)
                remaining -= 1
                if fj < fx:
                    x, fx = jump, fj
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
