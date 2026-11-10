"""Rewritten optimizer: stochastic line-search with golden-section refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic line-search with golden-section refinement for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: try origin and a few random points
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Evaluate a few random points to get a better starting point
        for _ in range(5):
            if calls >= budget:
                break
            cand = rng.uniform(-5, 5, dim)
            fc = func(cand)
            calls += 1
            if fc < fx:
                x, fx = cand, fc

        # Golden-section search along a direction
        def golden_search(x0, direction, calls_ref):
            """Find minimum of func along x0 + t*direction using golden-section.
            Returns (best_x, best_f, calls_used)."""
            # First, determine a reasonable range for t
            # Try expanding the range
            t_lo, t_hi = -1.0, 1.0
            f_lo = func(np.clip(x0 + t_lo * direction, -5, 5))
            f_hi = func(np.clip(x0 + t_hi * direction, -5, 5))
            calls_used = 2

            # Expand range if needed (up to 4 times)
            for _ in range(4):
                if calls_ref[0] + 4 > budget:
                    break
                t_new_hi = t_hi + (t_hi - t_lo)
                t_new_lo = t_lo - (t_hi - t_lo)
                f_new_hi = func(np.clip(x0 + t_new_hi * direction, -5, 5))
                f_new_lo = func(np.clip(x0 + t_new_lo * direction, -5, 5))
                calls_used += 2
                if f_new_hi < f_hi:
                    t_lo, f_lo = t_hi, f_hi
                    t_hi, f_hi = t_new_hi, f_new_hi
                elif f_new_lo < f_lo:
                    t_hi, f_hi = t_lo, f_lo
                    t_lo, f_lo = t_new_lo, f_new_lo
                else:
                    break

            # Golden-section iterations
            phi = (1 + np.sqrt(5)) / 2
            inv_phi = 1.0 / phi
            inv_phi2 = inv_phi ** 2

            # Interior points
            t1 = t_lo + inv_phi2 * (t_hi - t_lo)
            t2 = t_lo + inv_phi * (t_hi - t_lo)
            f1 = func(np.clip(x0 + t1 * direction, -5, 5))
            f2 = func(np.clip(x0 + t2 * direction, -5, 5))
            calls_used += 2

            # Iterate
            n_iter = min(20, budget - calls_ref[0] - calls_used)
            for _ in range(n_iter):
                if time.time() > deadline:
                    break
                if f1 < f2:
                    t_hi = t2
                    t2 = t1
                    f2 = f1
                    t1 = t_lo + inv_phi2 * (t_hi - t_lo)
                    f1 = func(np.clip(x0 + t1 * direction, -5, 5))
                else:
                    t_lo = t1
                    t1 = t2
                    f1 = f2
                    t2 = t_lo + inv_phi * (t_hi - t_lo)
                    f2 = func(np.clip(x0 + t2 * direction, -5, 5))
                calls_used += 1

            # Pick best
            if f1 < f2:
                best_t = t1
                best_f = f1
            else:
                best_t = t2
                best_f = f2

            best_x = np.clip(x0 + best_t * direction, -5, 5)
            calls_ref[0] += calls_used
            return best_x, best_f, calls_used

        # Main optimization loop
        # Start with standard basis directions + random directions
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
        # Add some random directions
        for _ in range(dim):
            d = rng.standard_normal(dim)
            d /= (np.linalg.norm(d) + 1e-12)
            directions.append(d)

        calls_ref = [calls]

        while calls_ref[0] < budget and time.time() < deadline:
            # Pick a random direction
            d = rng.standard_normal(dim)
            d /= (np.linalg.norm(d) + 1e-12)

            # Perform golden-section search along this direction
            best_x, best_f, n_calls = golden_search(x, d, calls_ref)

            if best_f < fx:
                x, fx = best_x, best_f
                # Update direction set: include the direction that worked
                # Keep top directions based on recent success
                if len(directions) > 2 * dim:
                    directions = directions[-2 * dim:]
                directions.append(d.copy())

            # Also try a few coordinate directions for local refinement
            if calls_ref[0] < budget and time.time() < deadline:
                for i in range(dim):
                    if calls_ref[0] >= budget or time.time() > deadline:
                        break
                    coord_dir = np.zeros(dim)
                    coord_dir[i] = 1.0
                    bx, bf, nc = golden_search(x, coord_dir, calls_ref)
                    if bf < fx:
                        x, fx = bx, bf

        return x, fx
# EVOLVE-BLOCK-END
