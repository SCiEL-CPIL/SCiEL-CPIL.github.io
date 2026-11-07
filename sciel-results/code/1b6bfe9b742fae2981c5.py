"""Combined (1+1)-ES with multi-start for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with proven adaptive step size and multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Multi-start: try several different starting points
        # Use deterministic seeds for reproducibility
        n_starts = 8
        budget_per_start = max(30, budget // n_starts)

        for start_idx in range(n_starts):
            if calls_made >= budget - 5:
                break
            if time.time() > deadline:
                break

            rng = np.random.default_rng(12345 + start_idx * 1000)

            # Different starting strategies:
            # 0: origin, 1-7: random in box
            if start_idx == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with proven adaptive step size
            # Success: sigma *= 1.5, Failure: sigma *= 0.87
            sigma = 2.0
            sigma_min = 1e-10
            max_steps = min(budget_per_start, budget - calls_made)

            for _ in range(max_steps):
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    break

                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87

                if sigma < sigma_min:
                    break

            # If we've used most of the budget, stop trying new starts
            if calls_made >= budget - 20:
                break

        # Final polish: fine-tune around the best point with very small steps
        if best_x is not None and calls_made < budget:
            if time.time() < deadline:
                rng = np.random.default_rng(99999)
                x = best_x.copy()
                fx = best_f
                sigma = 0.02
                remaining = min(50, budget - calls_made)

                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    if calls_made >= budget:
                        break

                    direction = rng.standard_normal(dim)
                    y = np.clip(x + sigma * direction, -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.3
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.85

                    if sigma < 1e-8:
                        break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            if calls_made < budget:
                best_f = func(x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
