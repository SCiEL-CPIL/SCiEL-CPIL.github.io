"""Coordinate-wise sequential search for Bent Cigar (f12) optimization."""
import numpy as np
import time


class Optimizer:
    """Coordinate-wise golden-section search for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0
        n = self.dim
        budget = self.budget

        # Golden ratio
        gr = (1 + np.sqrt(5)) / 2
        inv_gr = 1.0 / gr
        inv_gr2 = inv_gr * inv_gr

        def golden_section_1d(func, x, dim_idx, lo, hi, max_calls, calls_used):
            """Golden-section search along one dimension."""
            if hi - lo < 1e-10:
                return x, calls_used

            # Initial interior points
            x1 = hi - inv_gr * (hi - lo)
            x2 = lo + inv_gr * (hi - lo)

            f1 = self._eval_1d(func, x, dim_idx, x1, lo, hi)
            f2 = self._eval_1d(func, x, dim_idx, x2, lo, hi)
            calls_used += 2

            for _ in range(max_calls):
                if calls_used >= budget or time.time() > deadline:
                    break

                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - inv_gr * (hi - lo)
                    f1 = self._eval_1d(func, x, dim_idx, x1, lo, hi)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + inv_gr * (hi - lo)
                    f2 = self._eval_1d(func, x, dim_idx, x2, lo, hi)
                calls_used += 1

            # Best point is the one with lower function value
            if f1 < f2:
                best_val = x1
            else:
                best_val = x2

            x[dim_idx] = best_val
            return x, calls_used

        def _eval_1d(func, x, dim_idx, val, lo, hi):
            """Evaluate function with one dimension changed."""
            x_new = x.copy()
            x_new[dim_idx] = np.clip(val, lo, hi)
            return func(x_new)

        # Initialize with random points to find a good starting region
        best_x = None
        best_fx = np.inf
        calls_used = 0

        # Phase 1: Random sampling to find initial good point
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Coordinate-wise golden-section search
        # Cycle through dimensions multiple times with shrinking intervals
        x = best_x.copy()
        remaining = budget - calls_used

        # Number of cycles through all dimensions
        # Each cycle uses approximately n * 10 calls (10 golden-section steps per dim)
        # We want to use most of the budget
        calls_per_dim = 8  # golden-section iterations per dimension per cycle
        calls_per_cycle = n * (2 + calls_per_dim)  # 2 initial + calls_per_dim per dim
        max_cycles = max(1, remaining // calls_per_cycle)

        for cycle in range(max_cycles):
            if calls_used >= budget or time.time() > deadline:
                break

            # Shrink search interval based on cycle
            # Start with full range, then narrow around best_x
            if cycle == 0:
                search_radius = 5.0
            else:
                search_radius = max(0.1, 5.0 * (0.5 ** cycle))

            for dim_idx in range(n):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Define search interval around current best
                center = x[dim_idx]
                lo = max(-5.0, center - search_radius)
                hi = min(5.0, center + search_radius)

                if hi - lo < 1e-10:
                    continue

                x, calls_used = golden_section_1d(
                    func, x, dim_idx, lo, hi, calls_per_dim, calls_used
                )

                # Update best if improved
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 3: Fine-tuning with very small steps
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline - 5:
            # Do a few more passes with very small intervals
            for dim_idx in range(n):
                if calls_used >= budget or time.time() > deadline:
                    break
                small_radius = 0.05
                center = best_x[dim_idx]
                lo = max(-5.0, center - small_radius)
                hi = min(5.0, center + small_radius)
                if hi - lo < 1e-10:
                    continue
                x = best_x.copy()
                x, calls_used = golden_section_1d(
                    func, x, dim_idx, lo, hi, 5, calls_used
                )
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
