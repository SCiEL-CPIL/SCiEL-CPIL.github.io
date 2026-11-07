"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin for 300s limit
        
        # State tracking
        best_x = np.full(self.dim, np.nan)
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        # Wrapped function to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return 1e10  # Penalize to stop optimization
            # Clip to box to handle L-BFGS-B boundary issues gracefully
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            return val

        # Deterministic multi-start points
        # 1. Origin
        starts = [np.zeros(self.dim)]
        
        # 2. Coarse grid points (3^5 = 243 points is too many, use a subset or sparser grid)
        # Let's use a grid of -4, 0, 4 for each dimension? That's 3^5=243.
        # With 500 calls, we can afford roughly 10-20 L-BFGS runs if they converge fast.
        # Katsuura is multimodal but L-BFGS-B is a local method.
        # Let's try a set of strategic points.
        
        # Points: 0, and +/- 4 in each dimension combinations?
        # Maybe just a few random-like deterministic points and the origin.
        # Since the shift is hidden but within [-4,4], the origin is a good central guess.
        
        # Let's add points at +/- 4.5 (near boundaries) and +/- 2.5 (mid-range)
        # To ensure coverage, we'll generate a small set of diverse points.
        
        # 1. Origin
        # 2. Centered at +/- 4.0 in each dimension (too many: 2^5=32)
        # Let's use a quasi-random or grid-based approach.
        # Given 500 calls, and L-BFGS-B in 5D typically uses <50 calls per run,
        # we can do ~10 runs.
        
        # Let's pick 10 diverse starting points.
        # 0.0
        # +/- 4.0 in single dimensions
        # +/- 4.0 in two dimensions
        
        # Actually, a simpler robust approach for black-box with limited budget:
        # 1. Run L-BFGS-B from origin.
        # 2. If time/calls allow, run from other points.
        
        # However, Katsuura is known to be very multimodal. L-BFGS-B from origin might get stuck.
        # But the "shift" is applied. The global minimum is somewhere in [-4,4]^5.
        # A grid search of 3^5=243 points would cost 243 calls.
        # Then we could run L-BFGS-B from the best 5 grid points.
        # Total calls: 243 + 5*50 = 493. This fits the budget!
        
        # Strategy:
        # 1. Evaluate a grid of 3^5 = 243 points (values -4, 0, 4).
        # 2. Sort by function value.
        # 3. Run L-BFGS-B from the top 5 best points, until budget or time runs out.
        
        grid_vals = [-4.0, 0.0, 4.0]
        grid_points = list(np.meshgrid(*([grid_vals]*self.dim)))
        grid_points = np.array(grid_points).T.reshape(-1, self.dim)
        
        # Evaluate grid
        grid_fvals = np.empty(len(grid_points))
        for i, p in enumerate(grid_points):
            if calls_made >= max_calls:
                break
            grid_fvals[i] = wrapped_func(p)
        
        # If we ran out of calls during grid, return best found
        if calls_made >= max_calls:
            valid_indices = np.where(~np.isnan(grid_fvals))[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmin(grid_fvals[valid_indices])]
                best_x = grid_points[best_idx]
                best_f = grid_fvals[best_idx]
            return best_x, best_f

        # Find top 5 points
        valid_indices = np.arange(len(grid_points))
        top_k = 5
        sorted_indices = valid_indices[np.argsort(grid_fvals)]
        top_starts = grid_points[sorted_indices[:top_k]]
        
        # Run L-BFGS-B from these starts
        # We need to be careful with time.
        # L-BFGS-B can be slow if it doesn't converge.
        # Set maxiter to a reasonable limit, e.g., 20 iterations.
        # Each iteration in 5D is ~5 function calls (approx).
        # So 20 iters ~ 100 calls per run.
        # 5 runs * 100 = 500 calls. Might be tight.
        # Let's use maxiter=10 to be safe, and rely on the grid to find good basins.
        
        for start_point in top_starts:
            if calls_made >= max_calls:
                break
            if time.time() > deadline - 10.0: # 10s margin for final processing
                break
                
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    start_point, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)]*self.dim,
                    options={'maxiter': 15, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < best_f:
                    # Check if this is actually better
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except Exception:
                pass
            
            # If we are running low on calls, break
            if calls_made >= max_calls:
                break
                
        # Ensure best_x is not nan
        if np.isnan(best_x).any():
             # Fallback to first grid point
             best_x = grid_points[0]
             best_f = grid_fvals[0]
             
        return best_x, best_f
# EVOLVE-BLOCK-END
