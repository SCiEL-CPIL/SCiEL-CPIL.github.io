"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using Grid-Based Decomposition with Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 280  # 50s safety margin for 330s limit

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        # Helper to add a point if valid
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                # Clip to bounds just in case, though problem should do it
                x_clipped = np.clip(x, lo, hi)
                # Check if already added (simple check for exact match or close)
                # For efficiency in small lists, linear scan is fine
                for i in range(len(X_list)):
                    if np.allclose(X_list[i], x_clipped, atol=1e-8):
                        return
                X_list.append(x_clipped)
                F_list.append(f)
        
        # Phase 1: Grid-Based Coarse Search
        # With dim=4, a grid of size 4 per dimension gives 4^4 = 256 points.
        # Budget is 200. So we need to be careful.
        # Let's use a 3x3x3x3 grid = 81 points. This leaves 119 for refinement.
        # Or 4x4x4x2 = 128 points.
        # Let's stick to a uniform grid that fits the budget comfortably.
        # 200 budget. Let's reserve 100 for grid, 100 for local.
        # Grid size: We want to cover the space.
        # If we use 3 points per dimension: 3^4 = 81 evaluations.
        # This is efficient and provides good coverage.
        
        grid_size = 3
        n_grid = grid_size ** self.dim
        
        if n_grid > self.budget // 2:
             # Fallback to smaller grid if budget is tight, but 81 < 100 so it's fine
             grid_size = 2
             n_grid = grid_size ** self.dim

        # Generate grid points
        grid_axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(self.dim)]
        grid_points = np.array(np.meshgrid(*grid_axes)).T.reshape(-1, self.dim)
        
        # Evaluate grid points
        for x in grid_points:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
        
        if not F_list:
            # Fallback if grid failed completely
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0: add_point(x0, f0)
            return X_list if X_list else [x0]

        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Identify Promising Regions for Local Refinement
        # We want to refine points that are non-dominated or close to the Pareto front.
        # Simple heuristic: Select the top K points based on a scalarization 
        # (e.g., sum of normalized objectives) or just the non-dominated set.
        
        # Normalize objectives to [0, 1] based on current min/max for scalarization
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F - f_min) / f_range
        
        # Sum of normalized objectives (lower is better)
        sums = np.sum(F_norm, axis=1)
        
        # Select top candidates for refinement.
        # We have ~100-120 evals left. Nelder-Mead in 4D might take ~20-50 evals per run.
        # Let's try to run 4-6 local optimizations.
        n_refine = 5
        
        # Get indices of best candidates
        # To ensure diversity, we can pick the best from each "cluster" or just the best overall
        # Given the grid is uniform, picking the best few by sum is a good start.
        # However, weighted sum can miss parts of the front.
        # Let's pick a mix: best overall, and best in each "corner" of the normalized space?
        # Simpler: Just pick the top 5 non-dominated points.
        
        # Fast non-dominated sort for selection
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i+1, n):
                if is_dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        non_dom_indices = np.where(~is_dominated)[0]
        
        if len(non_dom_indices) == 0:
            # Fallback to best sum
            non_dom_indices = np.argsort(sums)[:n_refine]
        else:
            # If too many non-dominated, sample or pick best by sum among them
            if len(non_dom_indices) > n_refine * 2:
                 # Pick best n_refine * 2 by sum, then sample n_refine
                 sums_non_dom = sums[non_dom_indices]
                 best_non_dom = non_dom_indices[np.argsort(sums_non_dom)]
                 candidates = best_non_dom[:n_refine * 2]
                 # Random selection to ensure diversity in start points
                 rng = np.random.default_rng(42)
                 non_dom_indices = rng.choice(candidates, size=n_refine, replace=False)
            else:
                 # Use all non-dominated if few
                 non_dom_indices = non_dom_indices[:n_refine]
        
        # Phase 3: Local Refinement using Nelder-Mead
        for idx in non_dom_indices:
            if self.eval_count >= self.budget or self._check_time():
                break
            
            x_start = X[idx].copy()
            
            # Define objective for scalarization
            # Use a random weight vector to explore different parts of the front
            w = np.random.rand(self.n_obj)
            w /= np.sum(w)
            
            def objective(x):
                f = self._evaluate(x, problem)
                if f is None:
                    return 1e6
                return np.dot(w, f)
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'maxfev': 30, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # Evaluate the final result to get the vector
                if res.x is not None:
                    f_final = self._evaluate(res.x, problem)
                    if f_final is not None:
                        add_point(res.x, f_final)
                        
            except Exception:
                pass
        
        # Final cleanup: Return the list of X. The evaluator records F from its own calls?
        # "Every returned objective vector is recorded... return value of __call__ is ignored."
        # Wait, the contract says: "Every returned objective vector is recorded".
        # In the previous code, it returned X. The evaluator likely calls problem(x) again?
        # No, "problem(x) ... returns 3 objective values".
        # "Every returned objective vector is recorded" implies the evaluator records the output of `problem(x)` 
        # when it is called.
        # But the Optimizer calls `problem(x)` internally.
        # Does the evaluator intercept `problem`?
        # "evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "problem(x) clips ... and returns 3 objective values."
        # "Every returned objective vector is recorded".
        # This implies the `problem` passed to `__call__` is a wrapper that records.
        # So I just need to make sure I call `problem(x)` for every point I want recorded.
        # My `_evaluate` calls `problem(x)`. So all points added to X_list via `add_point` 
        # (which is only called after `_evaluate` returns a valid f) have been recorded.
        
        return X_list
# EVOLVE-BLOCK-END
