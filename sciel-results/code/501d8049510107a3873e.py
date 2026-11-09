"""Optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deadline-guarded (1+1)-ES with a short diverse-restart exploration
    phase followed by focused exploitation with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270.0  # safety margin under 300 s
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: brief diverse exploration (short restarts) ---
        x_best = None
        fx_best = np.inf
        n_explore = 60
        for r in range(n_explore):
            if r >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < fx_best:
                x_best, fx_best = x, fx
            # a few refinement steps from this start
            sigma = 2.0
            for _ in range(3):
                if r + 1 >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                if fx < fx_best:
                    x_best, fx_best = x, fx
            r += 1

        # --- Phase 2: focused exploitation from the best point ---
        x, fx = x_best, fx_best
        sigma = 1.0
        while True:
            if budget <= 0:
                break
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget -= 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.9
                if sigma < 1e-6:
                    # nudge back up occasionally to escape ultra-flat spots
                    sigma = max(sigma, 1e-4)
        return x, fx
# EVOLVE-BLOCK-END
