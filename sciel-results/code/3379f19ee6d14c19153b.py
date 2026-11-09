"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive (1+lambda)-ES with 1/5 success-rule step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 320s to leave a 10s safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: quick multi-start to locate the basin ---
        # Spend ~30% of budget on a coarse random search with a few local refinements
        n_init = min(60, budget // 4)
        best_x = None
        best_f = np.inf

        # Evaluate a set of random points, keep the best, then do a small local search from it
        init_points = rng.uniform(-5, 5, size=(n_init, dim))
        # Evaluate all init points
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = init_points[i]
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: Adaptive (1+lambda)-ES from the best point ---
        # Step size starts moderate; adapted via 1/5 rule over a window
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        lam = 4  # population size
        window = 20  # window length for 1/5 rule
        successes = []  # track successes in the window

        calls_used = n_init
        remaining = budget - calls_used

        for gen in range(remaining):
            if time.time() > deadline:
                break

            # Generate population
            # Use a mix: one step from current, plus some random perturbations
            # For robustness, generate lam offspring
            for j in range(lam):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Step size for this offspring
                if j == 0:
                    # First offspring: standard step
                    step = sigma * rng.standard_normal(dim)
                else:
                    # Other offspring: slightly larger exploration
                    step = (sigma * 1.2) * rng.standard_normal(dim)

                y = np.clip(x + step, -5, 5)
                fy = float(func(y))
                calls_used += 1

                # Track success
                success = fy < fx
                successes.append(success)
                if len(successes) > window:
                    successes.pop(0)

                if success:
                    x = y
                    fx = fy

            # Adapt sigma using 1/5 rule on the window
            if len(successes) >= 5:
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                # Keep sigma within reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
