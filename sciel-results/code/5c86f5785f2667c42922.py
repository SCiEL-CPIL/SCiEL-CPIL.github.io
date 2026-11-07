"""Direct-approach optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Diverse initialization – structured grid + random points – to
           cover the 21 basins of f22. The hidden rotation/shift makes a
           plain axis-aligned grid suboptimal, so we mix in random samples.
  Phase 2: Multi-start L-BFGS-B from the top diverse candidates for fast
           local convergence.
  Phase 3: If budget remains, jittered L-BFGS-B restarts around the best
           to escape shallow local minima.
  A wall-clock deadline guard (325 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Diverse-init then multi-start L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 325.0
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = np.asarray(x, dtype=float).copy()

        # ---------------- Phase 1: diverse initialization ----------------
        # Mix of structured grid points and random points to cover basins
        # regardless of hidden rotation/shift.
        rng = np.random.default_rng(12345)
        candidates = []

        # Structured grid: 3^5 = 243 points
        grid_vals = np.linspace(-5.0, 5.0, 3)
        grid_points = np.array(np.meshgrid(*([grid_vals] * n), indexing="ij")).reshape(-1, n)
        candidates.extend([gp.copy() for gp in grid_points])

        # Random points: fill remaining budget for phase 1
        # We want about 100 random points to diversify
        n_random = min(100, budget - len(candidates) - 50)
        if n_random > 0:
            rand_points = rng.uniform(-5.0, 5.0, size=(n_random, n))
            candidates.extend([rp.copy() for rp in rand_points])

        # Evaluate all candidates
        eval_results = []
        for cp in candidates:
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 10:
                break
            try:
                fc = fwrap(cp)
                eval_results.append((fc, cp.copy()))
                update_best(cp, fc)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0.copy()
            except Exception:
                return (None, np.inf)

        # Sort by function value
        eval_results.sort(key=lambda x: x[0])

        # ---------------- Phase 2: multi-start L-BFGS-B ----------------
        bounds = [(-5.0, 5.0) for _ in range(n)]
        # Take top 10 candidates for local optimization
        n_starts = min(10, len(eval_results))
        for i in range(n_starts):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 5:
                break
            start_x = eval_results[i][1]
            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 80, 'maxfun': 80})
                if res.fun < best["f"]:
                    update_best(res.x, res.fun)
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # ---------------- Phase 3: jittered restarts ----------------
        if best["x"] is not None and calls["count"] < budget - 10 and time.time() < deadline:
            for _ in range(8):
                if time.time() >= deadline:
                    break
                if calls["count"] >= budget - 5:
                    break
                jitter = rng.normal(0, 0.2, size=n)
                x_start = best["x"] + jitter
                x_start = np.clip(x_start, -5.0, 5.0)
                try:
                    res = minimize(fwrap, x_start, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 50, 'maxfun': 50})
                    if res.fun < best["f"]:
                        update_best(res.x, res.fun)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
