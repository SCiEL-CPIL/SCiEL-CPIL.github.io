"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive per-axis step sizes for sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit

        def safe_call(x):
            try:
                return func(np.asarray(x, dtype=float))
            except Exception:
                return np.inf

        # Initial point
        x = np.zeros(dim)
        fx = safe_call(x)

        # Per-axis adaptive step sizes
        steps = np.full(dim, 2.0)

        # Track best
        best_x = x.copy()
        best_fx = fx

        calls = 1

        # Main loop: coordinate descent with adaptive steps
        # We cycle through dimensions, trying to improve along each axis
        # If improvement, keep the new point and increase step; otherwise decrease

        dim_idx = 0
        consecutive_fails = 0

        while calls < budget:
            if time.time() > deadline:
                break

            d = dim_idx % dim

            # Try a step in positive direction
            x_try = x.copy()
            x_try[d] = np.clip(x_try[d] + steps[d], -5, 5)
            fx_try = safe_call(x_try)
            calls += 1

            if fx_try < fx:
                x = x_try
                fx = fx_try
                steps[d] *= 1.3  # Increase step on success
                consecutive_fails = 0
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Try negative direction
                x_try = x.copy()
                x_try[d] = np.clip(x_try[d] - steps[d], -5, 5)
                fx_try = safe_call(x_try)
                calls += 1

                if fx_try < fx:
                    x = x_try
                    fx = fx_try
                    steps[d] *= 1.3
                    consecutive_fails = 0
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                else:
                    # No improvement in either direction; shrink step
                    steps[d] *= 0.7
                    consecutive_fails += 1

            dim_idx += 1

            # If all axes have failed multiple times, try a random restart in a smaller region
            # around the current best to escape local flatness
            if consecutive_fails >= dim * 3:
                # Perturb around best with small random step
                rng = np.random.default_rng(calls)
                x_try = best_x + rng.standard_normal(dim) * 0.5
                x_try = np.clip(x_try, -5, 5)
                fx_try = safe_call(x_try)
                calls += 1
                if fx_try < best_fx:
                    x = x_try
                    fx = fx_try
                    best_x = x.copy()
                    best_fx = fx
                else:
                    x = best_x.copy()
                    fx = best_fx
                consecutive_fails = 0
                steps = np.full(dim, 1.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
