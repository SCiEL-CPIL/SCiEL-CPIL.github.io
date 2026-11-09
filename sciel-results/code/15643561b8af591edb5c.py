"""Grid-based global search followed by L-BFGS-B local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Coarse grid search to locate basin, then L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0

        call_count = 0
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Search
        # Use 3 points per dimension to cover [-3, 3]
        # This is 3^5 = 243 evaluations.
        # This leaves ~257 evaluations for polishing.
        grid_vals = [-3.0, 0.0, 3.0]
        grid_points = np.array(np.meshgrid(*[grid_vals] * dim)).T.reshape(-1, dim)
        
        # Evaluate all grid points
        grid_f = np.array([counted(p) for p in grid_points])
        
        # If we ran out of budget during grid search, return best
        if call_count >= budget or time.time() > deadline:
            if best_x is None:
                best_x = grid_points[np.argmin(grid_f)]
                best_f = np.min(grid_f)
            return best_x, best_f

        # Phase 2: L-BFGS-B Polish on top candidates
        # Select the 5 best points from the grid
        top_indices = np.argsort(grid_f)[:5]
        candidates = [grid_points[i].copy() for i in top_indices]
        
        # Ensure we have unique candidates
        unique_candidates = []
        for c in candidates:
            if not any(np.allclose(c, u, atol=1e-6) for u in unique_candidates):
                unique_candidates.append(c)
        
        # Add the current global best if not already there
        if best_x is not None:
            if not any(np.allclose(best_x, c, atol=1e-6) for c in unique_candidates):
                unique_candidates.append(best_x.copy())

        bounds = [(lo, hi)] * dim
        
        for cand in unique_candidates:
            if time.time() > deadline or call_count >= budget:
                break
            
            try:
                res = minimize(
                    counted,
                    cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # L-BFGS-B might stop early if budget is hit inside
            except Exception:
                pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
