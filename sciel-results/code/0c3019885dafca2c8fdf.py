"""Rewritten optimizer: multi-start Powell with LHS initialization, deadline-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell's method with Latin Hypercube initialization for Lunacek bi-Rastrigin f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Generate Latin Hypercube samples for diverse initialization
        # LHS ensures coverage of the search space
        n_lhs = 12
        lhs_points = []
        for i in range(n_lhs):
            # For each dimension, create a permutation of n_lhs intervals
            # and pick one point from each interval
            point = np.zeros(dim)
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                # Pick the i-th element of the permutation to determine which interval
                interval = perm[i]
                # Uniform within the interval
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        # Add some deterministic strategic points
        strategic_points = [
            np.zeros(dim),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
        ]

        # Combine and deduplicate-ish (just use all)
        all_starts = lhs_points + strategic_points

        # Phase 1: Evaluate all starting points
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: Powell's method from the best few starting points
        # Sort starts by their initial function value
        scored_starts = []
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            # We already evaluated these, but Powell will re-evaluate internally
            # Use the ones we found best
            scored_starts.append(x0)

        # Use top candidates for Powell
        # We'll run Powell from the global best and a few others
        powell_starts = [best_x.copy()] if best_x is not None else [np.zeros(dim)]
        # Add a few more diverse starts
        for x0 in all_starts[:6]:
            if x0 is not best_x and np.linalg.norm(x0 - best_x) > 1.0:
                powell_starts.append(x0.copy())
            if len(powell_starts) >= 4:
                break

        # Run Powell from each start
        for x_start in powell_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            try:
                res = minimize(
                    f,
                    x_start,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfev': 40}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: L-BFGS-B polish from the best point found by Powell
        if calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': 30}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If we still have budget, try a few more random restarts with Powell
        remaining_budget = budget - calls[0]
        if remaining_budget > 50 and time.time() < self._deadline:
            n_extra = min(3, remaining_budget // 50)
            for i in range(n_extra):
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                fx_rand = f(x_rand)
                if fx_rand < best_f:
                    best_f = fx_rand
                    best_x = x_rand.copy()
                try:
                    res = minimize(
                        f,
                        x_rand,
                        method='Powell',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 20, 'maxfev': 25}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
