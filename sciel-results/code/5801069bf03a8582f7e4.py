# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic adaptive coordinate descent (derandomized CMA-ES)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _scan_axis(self, func, x, fx, axis, n_pts, budget_left, deadline):
        """Sweep one coordinate; return (best_x, best_fx, calls_made)."""
        d = self.dim
        base = x[axis]
        old = x.copy()
        best_x = x.copy()
        best_fx = fx
        made = 0
        for k in range(n_pts):
            t = -5.0 + 10.0 * k / (n_pts - 1)
            if abs(t - base) < 1e-9:
                continue  # already have f at this value
            x[axis] = t
            f = float(func(x))
            made += 1
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
            if made >= budget_left:
                break
            if time.monotonic() > deadline:
                break
        # restore original axis value (keep best_x separate)
        x[axis] = base
        return best_x, best_fx, made

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.monotonic() + 295.0  # 5 s safety margin under 300 s

        # deterministic start (center of box)
        x = np.zeros(d)
        fx = float(func(x))
        best_x = x.copy()
        best_fx = fx
        calls = 1
        calls_left = budget - 1

        # resolution schedule: (n_pts_per_axis, max_cycles)
        schedule = [
            (7, 2),    # coarse, 2 global cycles
            (11, 1),
            (17, 1),
            (25, 1),
            (37, 1),
            (53, 1),
            (71, 1),
            (101, 1),
            (151, 1),
        ]

        # deterministic restart points (other basins) if fully stuck
        restarts = [
            np.zeros(d),
            np.full(d, 2.5),
            np.full(d, -2.5),
            np.array([2.5, -2.5, 2.5, -2.5, 2.5]),
            np.array([-2.5, 2.5, -2.5, 2.5, -2.5]),
            np.array([1.5, 1.5, -1.5, 1.5, -1.5]),
            np.array([-1.5, -1.5, 1.5, -1.5, 1.5]),
        ]
        restart_idx = 0

        for n_pts, max_cycles in schedule:
            for cycle in range(max_cycles):
                improved = False
                for axis in range(d):
                    if calls_left <= 0 or time.monotonic() > deadline:
                        break
                    x, fx, made = self._scan_axis(func, x, fx, axis, n_pts, calls_left, deadline)
                    calls += made
                    calls_left -= made
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        improved = True
                # if a full cycle made no progress, try a deterministic restart
                if not improved and calls_left > 0 and time.monotonic() < deadline:
                    if restart_idx < len(restarts):
                        x = restarts[restart_idx].copy()
                        restart_idx += 1
                        fx = float(func(x))
                        calls += 1
                        calls_left -= 1
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        # no more restarts; stop early
                        break
                if calls_left <= 0 or time.monotonic() > deadline:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
