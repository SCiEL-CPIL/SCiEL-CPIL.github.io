"""Direct-approach optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Coarse global grid (3^5 = 243 points) shuffled to ensure the
           best basin is likely encountered early, strongly improving AOCC.
  Phase 2: Focused L-BFGS-B refinement on the top 3 grid candidates for
           fast local convergence to the minimum.
  A wall-clock deadline guard (320 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-then-L-BFGS-B optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0
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

        # ---------------- Phase 1: coarse global grid ----------------
        # 3^5 = 243 points over [-5, 5]^5 (spacing ~2.5). This quickly locates
        # the best of the 21 basins, giving strong early AOCC progress.
        grid_n = 3
        grid_vals = np.linspace(-5.0, 5.0, grid_n)
        grid_points = np.array(np.meshgrid(*([grid_vals] * n), indexing="ij")).reshape(-1, n)

        # Shuffle the grid points to ensure that the best basin is likely
        # encountered earlier in the evaluation sequence, improving AOCC.
        rng = np.random.default_rng(42)
        perm = rng.permutation(grid_points.shape[0])
        grid_points = grid_points[perm]

        # Evaluate all grid points
        grid_results = []
        for gp in grid_points:
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 5:
                break
            gp = np.asarray(gp, dtype=float)
            try:
                fg = fwrap(gp)
                grid_results.append((fg, gp))
                update_best(gp, fg)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0.copy()
            except Exception:
                return (None, np.inf)

        # Sort grid results to get top candidates for local optimization
        grid_results.sort(key=lambda x: x[0])

        # ---------------- Phase 2: L-BFGS-B refinement ----------------
        # Focus on the top 3 grid candidates for efficient local convergence.
        # L-BFGS-B typically converges in < 50 iterations for 5D problems.
        bounds = [(-5.0, 5.0) for _ in range(n)]
        n_starts = min(3, len(grid_results))

        for i in range(n_starts):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget - 2:
                break
            start_x = grid_results[i][1]
            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'maxfun': 100})
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
            best["x"] = x0.copy()

        return best["x"], best["f"]
