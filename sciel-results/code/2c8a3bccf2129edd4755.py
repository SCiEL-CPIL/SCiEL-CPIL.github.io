"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """Global Screening + Brent's Method Coordinate Descent optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Coarse Global Screening
        # BBOB f7 (Step Ellipsoidal) has a step. We need to find the basin.
        # A coarse grid of 3^5 = 243 points is a good balance.
        # If budget allows, we evaluate this grid.
        grid_levels = 3
        grid_points = grid_levels ** dim
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        if budget >= grid_points + 10:
            coords = np.linspace(-5, 5, grid_levels)
            grid = np.array(np.meshgrid(*([coords]*dim)))
            grid = grid.reshape(dim, -1).T
            
            # Shuffle to randomize order (though symmetric, good practice)
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(grid))
            grid = grid[perm]
            
            for x in grid:
                if calls_made >= budget or time.time() > deadline:
                    break
                f_val = func(x)
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
        else:
            # Fallback: Random initialization
            rng = np.random.default_rng(42)
            pop_size = min(budget // 2, 20)
            population = rng.uniform(-5, 5, (pop_size, dim))
            for x in population:
                if calls_made >= budget or time.time() > deadline:
                    break
                f_val = func(x)
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

        if best_x is None:
            # Should not happen, but safety
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # 2. Local Refinement using Brent's Method Coordinate Descent
        # This is the core of the new strategy.
        # We perform cycles of 1D minimization along each axis.
        # Brent's method is very efficient (O(log N) function evaluations per axis).
        # This allows us to make many passes over the dimensions with the remaining budget.
        
        # We will continue until we run out of budget or time.
        # We use a shrinking step size for the 1D search intervals to ensure convergence.
        
        # Initial step size for the 1D search window.
        # Since the global basin is likely found, we can start with a moderate window.
        # However, to be safe against local minima or if the grid missed the center,
        # we start with a window that covers the local neighborhood.
        # Let's start with a window of size 1.0 around the current best.
        # We will shrink this window if we make progress, or keep it if we are stuck.
        
        # Actually, a simpler and robust approach for Coordinate Descent with Brent:
        # For each dimension i:
        #   Define f_1d(t) = func(best_x with best_x[i] = t)
        #   Minimize f_1d in [best_x[i] - delta, best_x[i] + delta]
        #   Update best_x[i] to the minimizer.
        
        # We need to handle the box constraints [-5, 5].
        # We clip the search interval to [-5, 5].
        
        # Let's use a fixed number of passes if budget allows, or until no improvement.
        # Since Brent is cheap, we can do many passes.
        # Let's estimate: 1 pass = 5 axes * ~10-20 calls/axis = 50-100 calls.
        # With 250 calls left (500 - 243 grid), we can do 2-5 passes.
        # This should be sufficient for high precision.
        
        # To maximize AOCC, we want to improve best_f quickly.
        # We will do passes until budget is exhausted or no significant improvement is seen.
        
        # We define a helper to minimize along one axis using Brent's method.
        def minimize_axis(x, axis, delta, calls_made, best_f_ref):
            # x: current point
            # axis: dimension to optimize
            # delta: half-width of the search interval
            # calls_made: current call count
            # best_f_ref: [best_f] to track improvement
            
            # Define the 1D function
            def f_1d(t):
                x_temp = x.copy()
                x_temp[axis] = t
                return func(x_temp)
            
            # Search interval
            lower = max(-5.0, x[axis] - delta)
            upper = min(5.0, x[axis] + delta)
            
            # If interval is too small, skip
            if upper - lower < 1e-8:
                return x, calls_made, best_f_ref[0]
            
            # Brent's method
            try:
                res = minimize_scalar(f_1d, bounds=(lower, upper), method='bounded', options={'xatol': 1e-8})
                
                # Update calls_made
                # res.nfev gives the number of function evaluations
                calls_made += res.nfev
                
                # Check time
                if time.time() > deadline:
                    return x, calls_made, best_f_ref[0]
                
                # Update best_x and best_f
                x_new = x.copy()
                x_new[axis] = res.x
                f_new = res.fun
                
                if f_new < best_f_ref[0]:
                    best_f_ref[0] = f_new
                    x[:] = x_new # In-place update
                    return x, calls_made, best_f_ref[0]
                else:
                    # No improvement, keep old x
                    return x, calls_made, best_f_ref[0]
            
            except Exception:
                # If Brent fails, skip this axis
                return x, calls_made, best_f_ref[0]

        # Main Coordinate Descent Loop
        # We use a list to track best_f for the helper function
        best_f_tracker = [best_f]
        
        # Initial delta for the search window
        # Start with a larger window to allow moving within the basin if the grid point was off-center.
        # Then shrink it.
        delta = 1.0
        
        # We will do passes until we run out of budget or time.
        # To avoid infinite loops with tiny improvements, we can stop if improvement is negligible.
        # But for AOCC, every bit of improvement counts.
        
        # Let's do a fixed number of passes based on remaining budget, or until no improvement.
        # Given Brent's efficiency, let's just loop while we have budget.
        
        # To ensure we make progress in early calls (AOCC), we should prioritize the first pass.
        # The first pass will likely yield significant improvement.
        
        while calls_made < budget and time.time() < deadline:
            # Check if we have enough budget for a full pass (5 axes * ~10 calls = 50 calls)
            # If not, we might break, but Brent might finish early.
            # Let's just try the next axis.
            
            # We iterate through dimensions in a fixed order (0, 1, ..., 4)
            # To avoid bias, we could shuffle the order each pass, but fixed is fine for CD.
            # Let's shuffle the order of axes each pass to avoid systematic bias.
            axes_order = np.random.permutation(dim)
            
            improved = False
            for axis in axes_order:
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Shrink delta slightly each pass to ensure convergence?
                # Or keep it constant?
                # If we keep it constant, we might oscillate.
                # If we shrink, we converge faster.
                # Let's shrink delta by 0.7 each full pass.
                
                old_best = best_f_tracker[0]
                best_x, calls_made, best_f_tracker[0] = minimize_axis(best_x, axis, delta, calls_made, best_f_tracker)
                
                if best_f_tracker[0] < old_best:
                    improved = True
            
            # If no improvement in a full pass, shrink the window and try again?
            # Or stop?
            # For f7, once in the basin, CD should converge.
            # If no improvement, it might be stuck in a local minimum or at the boundary.
            # Given the "Step" nature, if we are in the wrong basin, CD won't help.
            # But we assume the grid found the right basin.
            
            # Shrink delta for the next pass to refine
            delta *= 0.5
            
            # If delta is very small, we can stop
            if delta < 1e-6:
                break
            
            # Optional: If no improvement, break to save time for other strategies?
            # No, let's just break if no improvement to avoid wasting calls on a flat region.
            if not improved:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
