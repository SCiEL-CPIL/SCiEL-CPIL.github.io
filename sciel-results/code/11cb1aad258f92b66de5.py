"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find a good initial basin
        # Use diverse starting points: origin, axis-aligned points, and random
        starts = [np.zeros(d)]
        # Add axis-aligned points at moderate distances
        for i in range(d):
            for val in [-2.0, 2.0, -4.0, 4.0]:
                pt = np.zeros(d)
                pt[i] = val
                starts.append(pt)
        # Add some random points
        for _ in range(8):
            starts.append(rng.uniform(-5, 5, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: (1+1)-ES with adaptive sigma and periodic restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size
        no_improve_count = 0
        max_no_improve = 25  # restart if stuck for this many iterations

        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = try_eval(y)
            if fy is None:
                break

            if fy < fx:
                x, fx = y, fy
                no_improve_count = 0
                # Increase sigma on success (but cap it)
                sigma = min(sigma * 1.12, 4.0)
            else:
                no_improve_count += 1
                # Decrease sigma on failure
                sigma *= 0.88

            # Periodic restart if stuck
            if no_improve_count >= max_no_improve:
                # Try a new random start near current best
                new_x = np.clip(best_x + 1.5 * rng.standard_normal(d), -5, 5)
                new_fy = try_eval(new_x)
                if new_fy is None:
                    break
                if new_fy < fx:
                    x, fx = new_x, new_fy
                else:
                    # Keep current best, but reset sigma and counter
                    x = best_x.copy()
                    fx = best_f
                sigma = 0.8
                no_improve_count = 0

            # If sigma is very small, try a larger perturbation
            if sigma < 0.05:
                x = np.clip(x + 0.4 * rng.standard_normal(d), -5, 5)
                fx = try_eval(x)
                if fx is None:
                    break
                sigma = 0.4

        # Phase 3: Polish phase - use very small sigma for fine-tuning
        if calls_used < max_calls and time.time() < deadline - 3:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            polish_calls = min(30, max_calls - calls_used)
            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.7  # shrink further on success
                else:
                    sigma *= 0.6

        return best_x, best_f
# EVOLVE-BLOCK-END
