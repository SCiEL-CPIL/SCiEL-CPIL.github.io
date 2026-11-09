"""Deterministic coordinate-ascent optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Cyclic Coordinate Descent with Golden Section Search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def remaining():
            return budget - calls

        def time_ok():
            return time.time() < deadline

        # Phase 1: LHS warm-start for diverse coverage (80 calls)
        # Use a simple deterministic space-filling initialization
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        # LHS points: divide each dimension into n_bins and pick one point per bin
        n_points = min(79, budget - 1)  # leave at least 1 call
        if n_points > 0:
            n_bins = int(np.ceil(n_points ** (1.0 / dim)))
            n_per_bin = max(1, n_points // (n_bins ** dim))
            lhs_points = []
            for b in range(n_bins ** dim):
                # Convert linear index to multi-dimensional index
                idx = np.unravel_index(b, (n_bins,) * dim)
                # Random offset within each bin (deterministic seed)
                offset = np.random.RandomState(42 + b).random_sample(dim)
                pt = ((np.array(idx, dtype=float) + offset) / n_bins) * 10.0 - 5.0
                lhs_points.append(clip(pt))

            for pt in lhs_points:
                if calls >= budget - 1 or not time_ok():
                    break
                f_pt = func(pt)
                calls += 1
                if f_pt < best_f:
                    best_f = f_pt
                    best_x = pt.copy()

        # Phase 2: Cyclic Coordinate Descent with Golden Section Search
        # Start from the best point found
        x = best_x.copy()
        fx = best_f

        # Golden Section Search constants
        phi = (1 + np.sqrt(5)) / 2
        rho = 1 - 1 / phi  # = 1/phi

        # Number of GS iterations per line search
        gs_iters = 20

        # Initial step size for line search window
        # Start with a moderate step and adapt if needed
        step_scale = 1.0

        # We'll do multiple passes of cyclic coordinate descent
        # Each pass sweeps all dimensions
        # Stop when no improvement in a full pass or budget/time exhausted

        while remaining() > 10 and time_ok():
            improved_in_pass = False

            for i in range(dim):
                if remaining() < gs_iters + 2 or not time_ok():
                    break

                # Perform Golden Section Search along dimension i
                # Initial interval: [x[i] - step_scale, x[i] + step_scale]
                # Clip to box
                a = max(-5.0, x[i] - step_scale)
                b = min(5.0, x[i] + step_scale)

                if b - a < 1e-8:
                    continue

                # Golden Section Search
                # We need to find the minimum of f along dimension i
                # We'll evaluate at golden points
                # Initial interior points
                c = b - (b - a) / phi
                d = a + (b - a) / phi

                # Evaluate at c and d
                # Save original x
                x_orig = x[i]

                # Evaluate at c
                x_c = x.copy()
                x_c[i] = c
                x_c = clip(x_c)
                f_c = func(x_c)
                calls += 1

                # Evaluate at d
                x_d = x.copy()
                x_d[i] = d
                x_d = clip(x_d)
                f_d = func(x_d)
                calls += 1

                # If we've exceeded budget, break
                if remaining() < gs_iters:
                    # Still update best if improved
                    if f_c < fx:
                        x[i] = c
                        fx = f_c
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                            improved_in_pass = True
                    elif f_d < fx:
                        x[i] = d
                        fx = f_d
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                            improved_in_pass = True
                    break

                # GS loop
                for _ in range(gs_iters):
                    if remaining() < 2 or not time_ok():
                        break

                    if f_c < f_d:
                        # Minimum is in [a, d]
                        b = d
                        d = c
                        f_d = f_c
                        c = b - (b - a) / phi
                        x_c = x.copy()
                        x_c[i] = c
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls += 1
                    else:
                        # Minimum is in [c, b]
                        a = c
                        c = d
                        f_c = f_d
                        d = a + (b - a) / phi
                        x_d = x.copy()
                        x_d[i] = d
                        x_d = clip(x_d)
                        f_d = func(x_d)
                        calls += 1

                # After GS, the best point is either c or d (whichever has lower f)
                if f_c < f_d:
                    best_1d_pos = c
                    best_1d_val = f_c
                else:
                    best_1d_pos = d
                    best_1d_val = f_d

                # Update x if improved
                if best_1d_val < fx:
                    x[i] = best_1d_pos
                    fx = best_1d_val
                    improved_in_pass = True
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # No improvement along this axis; keep x[i] as is
                    # Optionally shrink step_scale for this axis
                    step_scale *= 0.7

            # If no improvement in a full pass, we're likely at a local minimum
            if not improved_in_pass:
                # Try a slightly wider search in case we're in a narrow valley
                step_scale = max(step_scale * 1.5, 0.1)
                if step_scale > 5.0:
                    break
                # One more pass
                continue
            else:
                # Had improvement; keep step_scale or shrink slightly
                step_scale *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
