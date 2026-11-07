"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+4)-ES with 1/5 success rule and periodic global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.0
        lam = 4
        success_hist = []
        max_hist = 5

        evals = 1
        last_explore_evals = 0
        explore_interval = 60  # Explore every 60 evaluations

        while evals < budget:
            if time.time() > deadline:
                break

            # Periodic global exploration: try a random restart if stuck
            if evals - last_explore_evals >= explore_interval:
                # Generate a random point far from current best
                x_try = rng.uniform(-5, 5, dim)
                f_try = func(x_try)
                evals += 1
                last_explore_evals = evals

                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    x = x_try.copy()
                    fx = f_try
                    sigma = 2.0
                    success_hist = []

                if evals >= budget:
                    break

            # Generate lam offspring
            ys = []
            for _ in range(lam):
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                ys.append(y)
                evals += 1
                if evals >= budget:
                    break

            if evals >= budget:
                break

            # Evaluate offspring
            fs = [func(y) for y in ys]

            # Find best offspring
            min_idx = np.argmin(fs)
            if fs[min_idx] < fx:
                x = ys[min_idx]
                fx = fs[min_idx]
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                success_hist.append(1)
            else:
                success_hist.append(0)

            # Trim history
            if len(success_hist) > max_hist:
                success_hist.pop(0)

            # Adapt sigma based on 1/5 rule
            if len(success_hist) >= 2:
                success_rate = sum(success_hist) / len(success_hist)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
