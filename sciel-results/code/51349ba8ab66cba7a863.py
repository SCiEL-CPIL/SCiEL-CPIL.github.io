"""Combined optimizer: LHS initialization + (1+1)-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube-style sampling for fast initial basin discovery
        n_lhs = min(30, budget // 15)
        best_x = None
        best_f = np.inf

        for i in range(n_lhs):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        calls_made = n_lhs

        while calls_made < budget:
            # Deadline check with safety margin
            if time.time() - start_time > 315.0:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
