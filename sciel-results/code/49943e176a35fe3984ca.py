"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with a coarse global scan."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._best_x = None
        self._best_f = float('inf')
        self._func = None

    def _counted_func(self, x):
        self._calls += 1
        return self._func(x)

    def __call__(self, func):
        self._func = func
        self._calls = 0
        self._best_x = None
        self._best_f = float('inf')
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Phase 1: coarse global scan with diverse starting points
        # Use a stratified grid + random points to cover the space
        n_start = min(40, budget // 15)  # leave room for L-BFGS-B
        starts = []

        # Stratified random points
        for _ in range(n_start):
            x0 = rng.uniform(-5, 5, dim)
            starts.append(x0)

        # Add some structured points (corners, center, axis-aligned)
        starts.append(np.zeros(dim))
        for i in range(dim):
            for val in [-4.0, 0.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate and limit
        seen = set()
        unique_starts = []
        for s in starts:
            key = tuple(np.round(s, 3))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        starts = unique_starts[:n_start]

        # Evaluate all starting points
        for x0 in starts:
            if self._calls >= budget or time.time() > deadline:
                break
            f0 = self._counted_func(x0)
            if f0 < self._best_f:
                self._best_f = f0
                self._best_x = x0.copy()

        # Phase 2: L-BFGS-B from the best few starting points
        # Sort starts by function value (we already have values for those evaluated)
        # Re-evaluate if needed, but we already have the best
        # Try L-BFGS-B from top candidates
        candidates = []
        # We have self._best_x, but let's also try a few others
        # Re-evaluate the top few starts to get their values
        for x0 in starts:
            if self._calls >= budget - 10 or time.time() > deadline:
                break
            # Check if we already evaluated this point
            # Simple approach: just evaluate and track
            f0 = self._counted_func(x0)
            candidates.append((f0, x0.copy()))
            if f0 < self._best_f:
                self._best_f = f0
                self._best_x = x0.copy()

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Run L-BFGS-B from the top candidates
        max_lbfgs_starts = min(5, len(candidates))
        for f0, x0 in candidates[:max_lbfgs_starts]:
            if self._calls >= budget - 5 or time.time() > deadline:
                break

            def local_obj(x):
                return self._counted_func(x)

            try:
                res = minimize(
                    local_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': min(100, budget - self._calls - 2),
                             'gtol': 1e-8}
                )
                if res.fun < self._best_f:
                    self._best_f = res.fun
                    self._best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final polish with a very tight L-BFGS-B from the best point
        if self._calls < budget - 2 and time.time() < deadline and self._best_x is not None:
            def local_obj(x):
                return self._counted_func(x)

            try:
                res = minimize(
                    local_obj,
                    self._best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'maxfun': min(50, budget - self._calls - 1),
                             'gtol': 1e-10}
                )
                if res.fun < self._best_f:
                    self._best_f = res.fun
                    self._best_x = res.x.copy()
            except Exception:
                pass

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
