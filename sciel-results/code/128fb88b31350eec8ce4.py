"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time-bounded exploration for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        calls = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls[0] += 1
            return float(func(x))

        # Phase 1: random exploration to find good basins
        rng = np.random.default_rng(42)
        n_explore = 200
        points = rng.uniform(-5, 5, (n_explore, self.dim))
        vals = np.empty(n_explore)
        for i in range(n_explore):
            if calls[0] >= self.budget or time.time() > deadline - 60:
                break
            vals[i] = wrapped(points[i])
        n_done = calls[0]

        # Select diverse good starting points
        if n_done > 0:
            valid = np.where(np.isfinite(vals[:n_done]))[0]
            if len(valid) > 0:
                v = vals[:n_done][valid]
                idx = valid
                # Sort by value, take top diverse points
                order = np.argsort(v)
                # Take top 15 by value, but ensure diversity
                starts = []
                used = set()
                for i in order[:60]:
                    if i in used:
                        continue
                    p = points[i]
                    # Check distance from existing starts
                    diverse = True
                    for sp in starts:
                        if np.linalg.norm(p - sp) < 1.5:
                            diverse = False
                            break
                    if diverse:
                        starts.append(p)
                        used.add(i)
                    if len(starts) >= 15:
                        break
                # If not enough diverse, fill with best
                if len(starts) < 15:
                    for i in order:
                        if i in used:
                            continue
                        starts.append(points[i])
                        used.add(i)
                        if len(starts) >= 15:
                            break
            else:
                starts = [points[0]]
        else:
            starts = [points[0]]

        # Phase 2: local refinement from each start
        best_x = None
        best_f = np.inf
        for sp in starts:
            if calls[0] >= self.budget - 5 or time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    wrapped,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 500, 'maxfun': 100, 'gtol': 1e-10},
                )
                if res.success or res.fun < best_f:
                    if np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                continue

        # Final local polish from best point
        if best_x is not None and calls[0] < self.budget - 2 and time.time() < deadline - 10:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
