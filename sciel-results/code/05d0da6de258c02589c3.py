"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard and stall restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before the 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        def elapsed_ok():
            return time.time() < deadline

        # --- Phase 1: short random exploration to seed a good starting point ---
        n_starts = min(8, max(1, self.budget // 10))
        best_x = None
        best_f = np.inf
        for _ in range(n_starts):
            if not elapsed_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        x = best_x
        fx = best_f
        sigma = 1.0

        calls_made = n_starts
        no_improve = 0
        max_no_improve = 40
        stall_restarts = 0
        max_stall_restarts = 6

        # --- Phase 2: (1+1)-ES with adaptive sigma and stall restarts ---
        while calls_made < self.budget:
            if not elapsed_ok():
                break

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
                no_improve = 0
            else:
                sigma *= 0.85
                no_improve += 1

            # Stall handling: try a bounded number of random restarts.
            if no_improve >= max_no_improve:
                improved = False
                for _ in range(min(3, self.budget - calls_made)):
                    if not elapsed_ok():
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
                        improved = True
                        break
                if improved:
                    no_improve = 0
                    sigma = 1.0
                else:
                    # No better restart found: shrink sigma and continue locally.
                    sigma *= 0.5
                    no_improve = 0
                    stall_restarts += 1
                    if stall_restarts >= max_stall_restarts:
                        break

            # Keep sigma in a workable range.
            if sigma < 1e-7:
                sigma = 0.3

        return best_x, best_f
# EVOLVE-BLOCK-END
