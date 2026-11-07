"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


class Optimizer:
    """Brent's Method-based Coordinate Descent optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Track function calls
        self._call_count = 0

        def f(x):
            self._call_count += 1
            return func(np.asarray(x, dtype=float))

        # Initial point: try a few random starts and pick the best
        n_init = min(15, budget // 10)
        x_best = None
        fx_best = np.inf
        for _ in range(n_init):
            x_cand = rng.uniform(-4, 4, d)
            fcand = f(x_cand)
            if fcand < fx_best:
                fx_best = fcand
                x_best = x_cand.copy()

        if self._call_count >= budget:
            return x_best, fx_best

        # Brent's method for 1D minimization on [lo, hi]
        def brent_1d(x0, axis, lo, hi, max_iter):
            """
            Minimize f along a single axis using Brent's method.
            x0: current point
            axis: which axis to minimize along
            lo, hi: bounds for the coordinate
            Returns (x_new, f_new)
            """
            x = x0.copy()
            coord = x[axis]

            # Define the 1D function
            def g(t):
                x[axis] = t
                return f(x)

            # Use scipy's minimize_scalar with method='bounded'
            res = minimize_scalar(g, bounds=(lo, hi), method='bounded',
                                   options={'maxiter': max_iter})
            x[axis] = res.x
            return x, res.fun

        # We'll do multiple passes of coordinate descent
        # Each pass goes through all dimensions
        # We allocate budget carefully

        # Estimate cost per coordinate descent step
        # Brent's method with bounded typically uses ~10-15 evaluations
        # Let's be conservative and use maxiter=8 for each 1D minimization
        max_iter_per_axis = 8

        # How many full passes can we afford?
        # Each pass costs at most d * max_iter_per_axis evaluations
        cost_per_pass = d * max_iter_per_axis
        n_passes = max(1, (budget - self._call_count) // cost_per_pass)

        # But we should also leave some budget for potential refinement
        # Let's do adaptive passes

        for p in range(n_passes):
            if self._call_count >= budget - 5:
                break

            for axis in range(d):
                if self._call_count >= budget - 5:
                    break

                # Determine bounds for this coordinate
                lo = -5.0
                hi = 5.0

                # Try to narrow the bounds using current best
                # Sample a few points to find a better bracket
                # Actually, let's just use the full box for simplicity
                # Brent's bounded method will handle it

                x_new, fx_new = brent_1d(x_best, axis, lo, hi, max_iter_per_axis)

                if fx_new < fx_best:
                    x_best = x_new
                    fx_best = fx_new
                else:
                    # If no improvement, keep the old point
                    pass

        # Final refinement: try a few more coordinate steps with smaller max_iter
        remaining = budget - self._call_count
        if remaining > 0:
            # Do a few more passes with reduced iterations
            small_max_iter = min(4, remaining // d)
            if small_max_iter > 0:
                for _ in range(min(2, remaining // (d * small_max_iter))):
                    for axis in range(d):
                        if self._call_count >= budget - 2:
                            break
                        x_new, fx_new = brent_1d(x_best, axis, -5.0, 5.0, small_max_iter)
                        if fx_new < fx_best:
                            x_best = x_new
                            fx_best = fx_new

        return x_best, fx_best
# EVOLVE-BLOCK-END
