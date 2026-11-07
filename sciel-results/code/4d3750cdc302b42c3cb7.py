# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start basin locator + (1+1)-ES refiner with a wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 280.0            # safety margin below the 300 s limit
        dim = self.dim

        # Deterministic, reproducible start (independent of instance).
        rng = np.random.default_rng(20240517)
        L, U = -5.0, 5.0

        # ---- Phase 1: multi-start basin locator (cheap, early AOCC progress) ----
        x = rng.uniform(L, U, dim)
        fx = float(func(x))
        n_calls = 1

        # A small number of exploratory restarts to find a low point / basin.
        for _ in range(12):
            if n_calls >= self.budget or time.time() - t0 > DEADLINE:
                break
            xc = rng.uniform(L, U, dim)
            fc = float(func(xc))
            n_calls += 1
            if fc < fx:
                x, fx = xc, fc

        # ---- Phase 2: (1+1)-ES refinement from the best basin ----
        sigma = 1.5
        best = (x.copy(), fx)
        remaining = self.budget - n_calls
        for _ in range(remaining):
            if time.time() - t0 > DEADLINE:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, L, U)
            fy = float(func(y))
            n_calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.35
                if fx < best[1]:
                    best = (x.copy(), fx)
            else:
                sigma *= 0.85
            if sigma < 1e-12:
                break

        if fx < best[1]:
            best = (x.copy(), fx)
        return best
# EVOLVE-BLOCK-END
