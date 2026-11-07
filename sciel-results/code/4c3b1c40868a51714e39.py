"""Rewritten optimizer: cyclic coordinate descent with golden-section line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GOLDEN = (5.0 ** 0.5 - 1.0) / 2.0  # ~0.618


class Optimizer:
    """Coordinate-descent minimizer with 1-D golden-section line searches."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300 s limit
        dim = self.dim
        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        # Start at the center of the box for the best initial Sphere error.
        x = np.zeros(dim, dtype=float)
        fx = f(x)

        # Golden-section 1-D minimization along axis `axis` of `x`.
        # Returns (new_x, new_f, n_calls_used).
        def line_search(x, fx, axis, remaining):
            # Build initial bracket around the current coordinate.
            lo = max(-5.0, x[axis] - 4.0)
            hi = min(5.0, x[axis] + 4.0)
            if hi - lo < 1e-12:
                return x, fx, 0

            a = lo
            b = hi
            # Standard golden-section interior points.
            c = b - _GOLDEN * (b - a)
            d = a + _GOLDEN * (b - a)

            xc = x.copy(); xc[axis] = c
            xd = x.copy(); xd[axis] = d
            if calls[0] + 2 > self.budget:
                return x, fx, 0
            fc = f(xc)
            fd = f(xd)
            n_used = 2

            # Iterate until interval is small or budget/time is exhausted.
            iters = 0
            max_iters = 60
            while (b - a) > 1e-6 and iters < max_iters:
                iters += 1
                if time.time() > deadline:
                    break
                if calls[0] + 1 > self.budget:
                    break
                if fc < fd:
                    # Minimum is in [a, d]
                    b = d
                    d = c
                    fd = fc
                    c = b - _GOLDEN * (b - a)
                    xc = x.copy(); xc[axis] = c
                    fc = f(xc)
                    n_used += 1
                else:
                    # Minimum is in [c, b]
                    a = c
                    c = d
                    fc = fd
                    d = a + _GOLDEN * (b - a)
                    xd = x.copy(); xd[axis] = d
                    fd = f(xd)
                    n_used += 1

            # Pick the best of the two interior points.
            if fc < fd:
                best_axis_val = c
                best_f = fc
            else:
                best_axis_val = d
                best_f = fd

            # Also compare against the starting point.
            if best_f < fx:
                new_x = x.copy()
                new_x[axis] = best_axis_val
                return new_x, best_f, n_used
            return x, fx, n_used

        # Cyclic coordinate descent with occasional randomized restarts.
        stagnation_threshold = 1e-10
        no_improve_passes = 0
        max_no_improve = 4

        # Pre-generate a few random restart points.
        rng = np.random.default_rng(12345)
        restarts = [rng.uniform(-5, 5, dim) for _ in range(6)]
        restart_idx = 0

        while calls[0] < self.budget and time.time() < deadline:
            fx_before = fx
            for axis in range(dim):
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                x, fx, _ = line_search(x, fx, axis, self.budget - calls[0])

            improvement = fx_before - fx
            if improvement < stagnation_threshold:
                no_improve_passes += 1
                if no_improve_passes >= max_no_improve and restart_idx < len(restarts):
                    # Try a random restart.
                    xr = restarts[restart_idx].copy()
                    restart_idx += 1
                    if calls[0] + 1 > self.budget:
                        break
                    fr = f(xr)
                    if fr < fx:
                        x, fx = xr, fr
                    no_improve_passes = 0
            else:
                no_improve_passes = 0

        return x, fx
# EVOLVE-BLOCK-END
