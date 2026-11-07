"""Optimizer for BBOB f11 (Discus) using Nelder-Mead with multi-restart."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart Nelder-Mead for smooth anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Wrap func to count calls and enforce budget
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped(x):
            if call_count[0] >= budget:
                return best_f[0]  # return best known to avoid extra calls
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # Generate diverse starting points
        # Use a combination of center, corners, and random points
        n_starts = 12
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Near-center random points
        for _ in range(4):
            starts.append(rng.uniform(-2, 2, dim))

        # Wider random points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        # Box edges
        for _ in range(2):
            starts.append(rng.uniform(-5, 5, dim))

        # One more random
        starts.append(rng.uniform(-5, 5, dim))

        # Sort starts by initial function value to prioritize good regions
        start_vals = []
        for s in starts:
            s = np.clip(s, -5.0, 5.0)
            fv = wrapped(s)
            start_vals.append((fv, s))

        start_vals.sort(key=lambda t: t[0])

        # Allocate remaining budget among restarts
        # Give more budget to better starting points
        remaining = budget - call_count[0]
        if remaining <= 0:
            return best_x[0], best_f[0]

        # Distribute budget: top starts get more iterations
        # Use a weighted scheme: better starts get more NM iterations
        weights = np.array([1.0 / (i + 1) for i in range(len(start_vals))])
        weights /= weights.sum()
        allocs = (weights * remaining).astype(int)
        # Ensure at least 10 iterations each if possible
        min_iter = 10
        for i in range(len(allocs)):
            if allocs[i] < min_iter and remaining > 0:
                allocs[i] = min_iter
                remaining -= 1
        # Adjust to fit exactly
        total_alloc = allocs.sum()
        diff = remaining - (budget - call_count[0] - total_alloc)
        # Just use what we have
        for i, (fv, s) in enumerate(start_vals):
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Set up Nelder-Mead
            x0 = s.copy()
            # Initial simplex: standard NM uses x0 + step*e_i
            # Use a moderate step size
            step = 0.5
            options = {
                'maxiter': allocs[i],
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            }

            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options=options,
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Final polish: if we have budget left, do a fine NM around best point
        if call_count[0] < budget and best_x[0] is not None and time.time() < deadline - 5.0:
            polish_budget = budget - call_count[0]
            x0 = best_x[0].copy()
            options = {
                'maxiter': polish_budget,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options=options,
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = wrapped(best_x[0])

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
