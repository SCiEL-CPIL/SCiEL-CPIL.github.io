"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with fast early exploration and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim

        # Start near center for fast initial progress
        x = np.zeros(dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        calls = 1

        # Adaptive step size, start large for fast early AOCC
        sigma = 3.0

        # Phase 1: (1+1)-ES with success rule, large initial steps
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.25, 8.0)
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.85

            # Phase 2: switch to smaller population search once close
            if sigma < 0.05 and i > 100:
                break

        # Phase 2: fine-grained local search with smaller steps
        if time.time() < deadline:
            sigma = max(sigma, 0.01)
            for i in range(self.budget - calls):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 0.5)
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
