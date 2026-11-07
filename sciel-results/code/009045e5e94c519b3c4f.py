"""Parallel coordinate descent with golden-section search for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Coordinate descent with golden-section search for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(42)

        def golden_section_1d(x, axis, lo, hi, max_iter=20):
            """Optimize a single coordinate using golden-section search."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x, 0

            # Golden ratio
            gr = (1.0 + math.sqrt(5.0)) / 2.0
            inv_gr = 1.0 / gr
            inv_gr2 = 1.0 / (gr * gr)

            # Initial points
            x1 = x.copy()
            x1[axis] = lo + inv_gr2 * (hi - lo)
            x2 = x.copy()
            x2[axis] = lo + inv_gr * (hi - lo)

            f1 = eval_and_track(x1)
            f2 = eval_and_track(x2)

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                if f1 < f2:
                    hi = x2[axis]
                    x2 = x1.copy()
                    f2 = f1
                    x1 = x.copy()
                    x1[axis] = lo + inv_gr2 * (hi - lo)
                    f1 = eval_and_track(x1)
                else:
                    lo = x1[axis]
                    x1 = x2.copy()
                    f1 = f2
                    x2 = x.copy()
                    x2[axis] = lo + inv_gr * (hi - lo)
                    f2 = eval_and_track(x2)

            # Return the better point
            if f1 < f2:
                return x1, f1
            else:
                return x2, f2

        # Phase 1: Coarse random search to find promising regions
        n_random = max(15, budget // 5)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Coordinate descent refinement on top candidates
        top_k = min(10, len(candidates))

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20:
                break

            # Coordinate descent: iterate over dimensions multiple times
            n_sweeps = max(1, min(5, remaining // (dim * 15)))

            for sweep in range(n_sweeps):
                if calls >= budget or time.time() > deadline:
                    break

                # Random coordinate order for this sweep
                order = rng.permutation(dim)

                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break

                    # Search range for this coordinate
                    lo = max(-5.0, x[axis] - 2.0)
                    hi = min(5.0, x[axis] + 2.0)

                    # If the coordinate is already at a boundary, adjust
                    if x[axis] <= -4.9:
                        lo = -5.0
                        hi = min(5.0, x[axis] + 1.0)
                    elif x[axis] >= 4.9:
                        lo = max(-5.0, x[axis] - 1.0)
                        hi = 5.0

                    x, _ = golden_section_1d(x, axis, lo, hi, max_iter=15)

            # Update best
            f_val = eval_and_track(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        # Phase 3: Fine-tuning around the best point found
        if calls < budget and time.time() < deadline:
            # Do a few more sweeps with tighter ranges
            x = best_x.copy()
            n_fine_sweeps = max(1, min(3, (budget - calls) // (dim * 10)))

            for sweep in range(n_fine_sweeps):
                if calls >= budget or time.time() > deadline:
                    break

                order = rng.permutation(dim)
                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break

                    lo = max(-5.0, x[axis] - 0.5)
                    hi = min(5.0, x[axis] + 0.5)

                    x, _ = golden_section_1d(x, axis, lo, hi, max_iter=10)

            f_val = eval_and_track(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        # Phase 4: If budget remains, do random restarts from best point
        if calls < budget and time.time() < deadline:
            step = 0.5
            while calls < budget and time.time() < deadline:
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    step = max(0.1, step * 0.8)
                else:
                    step = min(2.0, step * 1.2)

        return best_x, best_f
# EVOLVE-BLOCK-END
