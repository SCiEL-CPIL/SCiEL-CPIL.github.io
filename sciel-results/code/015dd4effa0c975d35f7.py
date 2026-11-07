"""Direct-approach optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Coarse global grid (3^5 = 243 points) to locate the best basin
           early, which strongly improves the AOCC area-under-curve score.
  Phase 2: L-BFGS-B refinement centered on the best grid point for fast
           local convergence to the minimum.
  A wall-clock deadline guard (330 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-then-L-BFGS-B optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0  # safety margin before 330 s kill
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
                best["x"] = x.copy()

        # ---------------- Phase 1: coarse global grid ----------------
        # 3^5 = 243 points over [-5, 5]^5 (spacing ~2.5). This quickly locates
        # the best of the 21 basins, giving strong early AOCC progress.
        grid_n = 3
        grid_vals = np.linspace(-5.0, 5.0, grid_n)
        grid_points = list(np.meshgrid(*([grid_vals] * n), indexing="ij"))
        grid_points = np.array([p for p in zip(*grid_points)])
        grid_points = np.asarray(grid_points).reshape(-1, n)

        # Shuffle the grid points to ensure that the best basin is likely
        # encountered earlier in the evaluation sequence, improving AOCC.
        # Use a fixed seed for reproducibility of the search strategy.
        rng = np.random.default_rng(42)
        perm = rng.permutation(grid_points.shape[0])
        grid_points = grid_points[perm]

        # We can afford to evaluate all 243 points as 243 < 500
        grid_results = []
        for gp in grid_points:
            if time.time() >= deadline:
                break
            gp = np.asarray(gp, dtype=float)
            try:
                fg = fwrap(gp)
                grid_results.append((fg, gp))
                update_best(gp, fg)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            # Fallback if grid failed immediately
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0
            except:
                return (None, np.inf)

        # Sort grid results to get top candidates for local optimization
        grid_results.sort(key=lambda x: x[0])
        
        # ---------------- Phase 2: L-BFGS-B refinement ----------------
        # Use L-BFGS-B to converge quickly in the best basins.
        # We will try the top few grid points as starting locations.
        
        # Reserve some budget for potential multiple restarts if the first one is fast
        # L-BFGS-B typically converges in < 100 iterations for 5D.
        
        candidates = []
        for i in range(min(5, len(grid_results))):
            candidates.append(grid_results[i][1])
        
        # If we have budget left, we can try multiple starts
        # The cost of L-BFGS-B is proportional to number of function calls.
        # Let's try to run L-BFGS-B on the best candidate first.
        
        for start_x in candidates:
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break
            
            # L-BFGS-B with bounds
            bounds = [(-5.0, 5.0) for _ in range(n)]
            
            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'maxfun': 100})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If L-BFGS-B fails for some other reason, continue to next candidate
                continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
