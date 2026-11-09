"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hierarchical coordinate-descent with golden-section search per coordinate,
    combined with a global random-restart safety net.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill

        def time_left():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Golden section search along one coordinate of x
        def golden_search_coord(x, coord_idx, lo, hi, x_best, f_best):
            nonlocal calls_made, best_x, best_f
            if not time_left():
                return x, f_best
            # We will do at most ~20 function evaluations for this coordinate
            # to keep total budget reasonable
            phi = (1 + np.sqrt(5)) / 2
            invphi2 = (2 - np.sqrt(5)) / 2  # ~ -0.618
            # Actually standard golden section:
            # c = hi - (hi-lo)/phi, d = lo + (hi-lo)/phi
            # Let's use the standard constants
            gr = (np.sqrt(5) - 1) / 2  # 0.618...
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)

            x_c = x.copy()
            x_c[coord_idx] = c
            f_c = func(x_c)
            calls_made += 1
            if f_c < f_best:
                best_x, best_f = x_c.copy(), f_c

            x_d = x.copy()
            x_d[coord_idx] = d
            f_d = func(x_d)
            calls_made += 1
            if f_d < f_best:
                best_x, best_f = x_d.copy(), f_d

            for _ in range(18):  # max iterations
                if not time_left():
                    break
                if f_c < f_d:
                    hi = d
                    d = c
                    f_d = f_c
                    c = hi - gr * (hi - lo)
                    x_c = x.copy()
                    x_c[coord_idx] = c
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < f_best:
                        best_x, best_f = x_c.copy(), f_c
                else:
                    lo = c
                    c = d
                    f_c = f_d
                    d = lo + gr * (hi - lo)
                    x_d = x.copy()
                    x_d[coord_idx] = d
                    f_d = func(x_d)
                    calls_made += 1
                    if f_d < f_best:
                        best_x, best_f = x_d.copy(), f_d

            # Final midpoint check
            mid = (lo + hi) / 2
            if abs(mid - x[coord_idx]) > 1e-6:
                x_mid = x.copy()
                x_mid[coord_idx] = mid
                f_mid = func(x_mid)
                calls_made += 1
                if f_mid < f_best:
                    best_x, best_f = x_mid.copy(), f_mid
                    x[coord_idx] = mid
                    f_best = f_mid
            return x, f_best

        # Coordinate descent pass: iterate through all dimensions
        def coord_descent_pass(x, f_x, max_passes=3):
            nonlocal calls_made, best_x, best_f
            for p in range(max_passes):
                if not time_left() or calls_made >= budget - 10:
                    break
                for i in range(dim):
                    if not time_left() or calls_made >= budget - 10:
                        break
                    # Search in a local window around current value
                    # Use a window that shrinks as we converge
                    # Start with full range, then narrow
                    window = 2.0 * (0.5 ** p)  # 2.0, 1.0, 0.5
                    lo = max(-5.0, x[i] - window)
                    hi = min(5.0, x[i] + window)
                    if hi - lo < 1e-6:
                        continue
                    x, f_x = golden_search_coord(x, i, lo, hi, x, f_x)
            return x, f_x

        # Main loop: multiple random restarts with coordinate descent
        rng = np.random.default_rng(42)

        # Initial point: try center first, then random
        # Actually, let's do a few random starts
        n_restarts = 5
        per_restart_budget = max(10, (budget - 5) // n_restarts)

        for r in range(n_restarts):
            if not time_left() or calls_made >= budget - 5:
                break

            # Choose starting point
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            f_x = func(x)
            calls_made += 1
            if f_x < best_f:
                best_x, best_f = x.copy(), f_x

            # Do coordinate descent from this start
            remaining = per_restart_budget - 1
            if remaining <= 0:
                continue

            # Adaptive number of passes based on remaining budget
            # Each pass uses ~dim * 20 evaluations roughly, but we cap
            # Let's just do passes until budget or time runs out
            for p in range(10):  # max 10 passes
                if not time_left() or calls_made >= budget - 5:
                    break
                for i in range(dim):
                    if not time_left() or calls_made >= budget - 5:
                        break
                    # Window shrinks with passes
                    window = 2.5 * (0.6 ** p)
                    lo = max(-5.0, x[i] - window)
                    hi = min(5.0, x[i] + window)
                    if hi - lo < 1e-5:
                        continue
                    x, f_x = golden_search_coord(x, i, lo, hi, x, f_x)

        # If we have budget left, do a final fine-tuning pass on best_x
        if best_x is not None and time_left() and calls_made < budget - 2:
            x = best_x.copy()
            f_x = best_f
            for p in range(5):
                if not time_left() or calls_made >= budget - 2:
                    break
                for i in range(dim):
                    if not time_left() or calls_made >= budget - 2:
                        break
                    window = 0.5 * (0.5 ** p)
                    lo = max(-5.0, x[i] - window)
                    hi = min(5.0, x[i] + window)
                    if hi - lo < 1e-6:
                        continue
                    x, f_x = golden_search_coord(x, i, lo, hi, x, f_x)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
