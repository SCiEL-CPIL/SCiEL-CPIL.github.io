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
        # The BBOB f7 (Step Ellipsoidal) has a 'step' in the objective function.
        # Global exploration is needed to find the region containing the global minimum.
        # We perform a coarse grid search to identify the best basin.
        
        # Grid size: 3 levels per dimension -> 3^5 = 243 points.
        # This fits well within the budget of 500 and provides a reasonable initial spread.
        grid_levels = 3
        grid_points = grid_levels ** dim
        
        rng = np.random.default_rng(42)
        
        if budget >= grid_points + 10:
            coords = np.linspace(-5, 5, grid_levels)
            grid = np.array(np.meshgrid(*([coords]*dim)))
            grid = grid.reshape(dim, -1).T
            # Shuffle to avoid bias
            perm = rng.permutation(len(grid))
            grid = grid[perm]
            
            # Evaluate grid
            fvals = np.empty(len(grid))
            calls_made = 0
            for i in range(len(grid)):
                if calls_made >= budget:
                    break
                fvals[i] = func(grid[i])
                calls_made += 1
            
            # Find best point in grid
            best_idx = np.argmin(fvals[:calls_made])
            best_x = grid[best_idx].copy()
            best_f = fvals[best_idx]
        else:
            # Fallback: Random initialization
            pop_size = min(budget // 2, 20)
            population = rng.uniform(-5, 5, (pop_size, dim))
            fvals = np.empty(pop_size)
            calls_made = 0
            for i in range(pop_size):
                fvals[i] = func(population[i])
                calls_made += 1
            best_idx = np.argmin(fvals)
            best_x = population[best_idx].copy()
            best_f = fvals[best_idx]

        # 2. Local Refinement using Coordinate Descent with Brent's Method
        # The Step Ellipsoidal function is separable in its local coordinate system.
        # Even with rotation, coordinate descent with a good 1D solver (Brent) is very effective
        # for ellipsoidal-like functions, especially when the condition number isn't extreme.
        # We iterate through dimensions, minimizing along each axis using Brent's method.
        
        # Helper function to minimize along one dimension
        def minimize_axis(x_current, axis, max_calls, func, calls_made, best_x, best_f):
            # Define the 1D function
            def f_1d(t):
                x_new = x_current.copy()
                x_new[axis] = t
                return func(x_new)
            
            # Determine search interval for Brent's method
            # We look in a window around the current point.
            # The window size should be large enough to capture the minimum if it's nearby.
            # Since the box is [-5, 5], and we are likely in the basin, a window of +/- 2 or 3 is reasonable.
            # However, if the current point is near the boundary, we must clip.
            
            center = x_current[axis]
            # Adaptive window size: start with 1.0, expand if needed? 
            # For simplicity, let's use a fixed window of 1.0 initially, 
            # but ensure it stays within [-5, 5].
            # Actually, since the grid step was ~3.3, the minimum could be up to 1.65 away.
            # Let's use a window of 2.0 to be safe.
            half_window = 2.0
            a = max(-5.0, center - half_window)
            b = min(5.0, center + half_window)
            
            # If the window is degenerate (a == b), we can't minimize.
            if b - a < 1e-6:
                return calls_made, best_x, best_f, False
            
            # Use scipy.optimize.minimize_scalar with 'brent' method
            # We need to count function calls. 
            # Since minimize_scalar doesn't directly expose the call count in a simple callback that stops it,
            # we have to be careful. 
            # However, Brent's method typically converges in ~10-20 evaluations.
            # We can wrap the function to count calls and raise an exception if we exceed budget?
            # Or just trust that it will be fast. 
            # Given the deadline check, we can also check time.
            
            # To be safe with the budget, let's implement a simple manual Brent-like search or
            # use minimize_scalar but with a tight maxiter.
            # minimize_scalar(method='brent', bounds=(a, b), maxiter=50)
            
            # Let's use a wrapper that counts calls and respects the global deadline/budget.
            local_calls = 0
            def f_wrapper(t):
                nonlocal local_calls, calls_made, best_f, best_x
                if calls_made >= max_calls or time.time() > deadline:
                    raise TimeoutError("Budget or time exceeded")
                
                val = f_1d(t)
                local_calls += 1
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x_current.copy()
                    best_x[axis] = t
                return val
            
            try:
                res = minimize_scalar(f_wrapper, bounds=(a, b), method='brent', options={'xatol': 1e-10, 'maxiter': 50})
                # Update best_x and best_f with the result
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = x_current.copy()
                    best_x[axis] = res.x
                elif res.success:
                    # Even if not better than global best, update local best for this axis step?
                    # Actually, we want to return the new x for the next axis.
                    # So we should update x_current's axis with res.x
                    x_current[axis] = res.x
            except TimeoutError:
                # If we ran out of budget/time, we just return current state
                pass
            except Exception:
                # Catch any other errors from minimize_scalar
                pass

            # If we didn't hit the limit, we might want to do another pass with a smaller window?
            # Or just return.
            return calls_made, best_x, best_f, True

        # Coordinate Descent Loop
        # We iterate through all dimensions multiple times until budget is exhausted or convergence.
        # Each pass consists of dim 1D minimizations.
        
        # We have budget - calls_made left.
        # Each 1D minimization takes ~10-20 calls.
        # So we can do roughly (budget - calls_made) / (dim * 15) passes.
        
        # Let's just run the loop and let the budget/deadline stop it.
        while calls_made < budget and time.time() < deadline:
            improved = False
            for axis in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Call minimize_axis
                calls_made, best_x, best_f, success = minimize_axis(best_x, axis, budget, func, calls_made, best_x, best_f)
                if success:
                    improved = True
            
            # If a full pass didn't improve (or significantly improve), we might stop?
            # But for AOCC, we want to keep improving as long as we can.
            # Brent's method is very efficient, so we should keep going until budget is spent.
            # However, if the function value stops changing, we are at the minimum.
            # We can check the change in best_f.
            # Let's add a simple convergence check: if best_f didn't change in the last pass, break.
            # (This is optional, as Brent is fast and we might just waste a few calls on flat regions)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
