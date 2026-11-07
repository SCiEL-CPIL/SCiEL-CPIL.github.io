"""Optimizer for BBOB-style Rastrigin (f3) with integer-aware lattice search."""
import numpy as np
import time


class Optimizer:
    """Integer-aware lattice search + local refinement for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return True

        # Phase 1: Integer lattice search
        # Rastrigin has global minima at integer coordinates (after shift/rotation)
        # Sample all integer points in [-4, 4]^5
        int_range = range(-4, 5)  # -4 to 4 inclusive
        for i0 in int_range:
            for i1 in int_range:
                for i2 in int_range:
                    for i3 in int_range:
                        for i4 in int_range:
                            if not try_point(np.array([i0, i1, i2, i3, i4], dtype=float)):
                                break
                        if not try_point(np.array([i0, i1, i2, i3, i4], dtype=float)):
                            break
                    if not try_point(np.array([i0, i1, i2, i3, i4], dtype=float)):
                        break
                if not try_point(np.array([i0, i1, i2, i3, i4], dtype=float)):
                    break
            if not try_point(np.array([i0, i1, i2, i3, i4], dtype=float)):
                break

        # Phase 2: Near-integer offsets around best found point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Try small offsets around the best integer point
            best_int = np.round(best_x).astype(int)
            offsets = np.array([
                [0, 0, 0, 0, 0],
                [0.1, 0, 0, 0, 0], [-0.1, 0, 0, 0, 0],
                [0, 0.1, 0, 0, 0], [0, -0.1, 0, 0, 0],
                [0, 0, 0.1, 0, 0], [0, 0, -0.1, 0, 0],
                [0, 0, 0, 0.1, 0], [0, 0, 0, -0.1, 0],
                [0, 0, 0, 0, 0.1], [0, 0, 0, 0, -0.1],
                [0.2, 0, 0, 0, 0], [-0.2, 0, 0, 0, 0],
                [0, 0.2, 0, 0, 0], [0, -0.2, 0, 0, 0],
                [0, 0, 0.2, 0, 0], [0, 0, -0.2, 0, 0],
                [0, 0, 0, 0.2, 0], [0, 0, 0, -0.2, 0],
                [0, 0, 0, 0, 0.2], [0, 0, 0, 0, -0.2],
                [0.3, 0, 0, 0, 0], [-0.3, 0, 0, 0, 0],
                [0, 0.3, 0, 0, 0], [0, -0.3, 0, 0, 0],
                [0, 0, 0.3, 0, 0], [0, 0, -0.3, 0, 0],
                [0, 0, 0, 0.3, 0], [0, 0, 0, -0.3, 0],
                [0, 0, 0, 0, 0.3], [0, 0, 0, 0, -0.3],
            ])
            for off in offsets:
                if not try_point(best_int.astype(float) + off):
                    break

        # Phase 3: Local refinement with (1+1)-ES from best point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 0.01:
                        break

        # Phase 4: If we have budget left, do a finer lattice around best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            best_int = np.round(best_x).astype(int)
            # Fine grid with step 0.05 in a box around best_int
            fine_range = np.arange(-0.5, 0.55, 0.05)
            for d0 in fine_range:
                for d1 in fine_range:
                    for d2 in fine_range:
                        for d3 in fine_range:
                            for d4 in fine_range:
                                pt = best_int.astype(float) + np.array([d0, d1, d2, d3, d4])
                                if not try_point(pt):
                                    break
                            if not try_point(pt):
                                break
                        if not try_point(pt):
                            break
                    if not try_point(pt):
                        break
                if not try_point(pt):
                    break
            if not try_point(pt):
                break

        return best_x, best_f
