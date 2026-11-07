"""Rewritten optimizer for BBOB-style Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """
    CMA-ES-like optimizer with adaptive step size, restarts, and time budget.
    Designed for Rastrigin (multimodal) on [-5, 5]^5 with 500 function calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 295.0  # safety margin before 300s
        rng = np.random.default_rng(12345)

        dim = self.dim
        n = self.budget
        x_best = None
        f_best = np.inf
        calls_used = 0

        # Number of restarts
        # For 500 calls, maybe 5-10 restarts of ~50-100 calls each
        n_restarts = 8
        budget_per_restart = n // n_restarts
        # Keep some slack for final polish
        # Let's do: 7 restarts of ~50, then 1 final phase with remaining budget
        # Actually, let's be more dynamic

        for restart in range(n_restarts):
            if time.time() > self.deadline:
                break

            # Initialize around random point or around best found so far
            if x_best is None:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb best
                sigma_init = 1.0
                x0 = np.clip(x_best + rng.standard_normal(dim) * sigma_init, -5, 5)

            # Run a short CMA-ES-like loop
            x = x0.copy()
            fx = func(x)
            calls_used += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

            # Adapt sigma based on initial success
            sigma = 0.5
            successes = 0
            total = 0
            max_iter_this_restart = min(budget_per_restart - 1, n - calls_used)

            for i in range(max_iter_this_restart):
                if time.time() > self.deadline:
                    break
                if calls_used >= n:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                total += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    # Increase sigma slightly on success
                    sigma *= 1.2
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85

                # Adaptive sigma update every 10 steps
                if total >= 10:
                    rate = successes / total
                    # 1/5 rule
                    if rate > 0.2:
                        sigma *= 1.1
                    elif rate < 0.2:
                        sigma *= 0.9
                    successes = 0
                    total = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Final local search around best point if budget remains
        if x_best is not None and calls_used < n:
            x = x_best.copy()
            fx = f_best
            sigma = 0.1
            for i in range(n - calls_used):
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x = y
                    fx = fy
                    f_best = fx
                    x_best = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-8, 1.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
