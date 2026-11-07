"""Rewritten optimizer for BBOB-style Rastrigin (f3) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive (1+1)-ES with multiple restarts and momentum for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            return float(func(x))

        def local_search(x0, f0, max_calls):
            nonlocal calls_used, best_x, best_f

            x = x0.copy()
            fx = f0
            sigma = 1.0
            momentum = np.zeros(dim)
            beta = 0.7

            for _ in range(max_calls):
                if calls_used >= budget - 1:
                    break

                # Adaptive step size based on success rate (simple version)
                y = x + sigma * (momentum + rng.standard_normal(dim))
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                if fy < fx:
                    # Success: update momentum and increase sigma
                    momentum = beta * momentum + (1 - beta) * (y - x)
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.85

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

            return x, fx

        # Phase 1: Broad exploration with multiple random starts
        n_exploration_starts = 10
        for i in range(n_exploration_starts):
            if calls_used >= budget - 50:
                break

            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-2, 2, dim)
            elif i == 2:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            f0 = safe_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Local search from best point
        if best_x is not None and calls_used < budget - 10:
            remaining = budget - calls_used
            local_search(best_x, best_f, remaining)

        # Phase 3: If budget remains, try a few more random restarts
        if calls_used < budget - 20:
            for _ in range(3):
                if calls_used >= budget - 10:
                    break
                x0 = rng.uniform(-5, 5, dim)
                f0 = safe_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                    if calls_used < budget - 10:
                        local_search(best_x, best_f, min(30, budget - calls_used - 5))

        return best_x, best_f
# EVOLVE-BLOCK-END
