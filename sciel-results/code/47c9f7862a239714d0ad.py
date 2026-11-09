# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global random search to locate the global basin, then local (1+1)-ES to refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin well under 330s
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: global random search to find a good basin
        n_global = max(10, budget // 10)
        n_global = min(n_global, budget - 1)

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        n_calls = 1

        for _ in range(n_global):
            if time.time() > deadline:
                break
            y = rng.uniform(-5, 5, dim)
            fy = func(y)
            n_calls += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

        # Phase 2: local (1+1)-ES refinement from the global best
        x = best_x.copy()
        fx = best_fx
        sigma = 1.5
        remaining = budget - n_calls

        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            n_calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            else:
                sigma *= 0.85

        return best_x, best_fx
# EVOLVE-BLOCK-END
