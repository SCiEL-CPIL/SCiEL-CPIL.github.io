"""Rewritten optimizer: coarse-to-fine 1D grid search exploiting Rastrigin separability, then Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine 1D coordinate search + Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        # Budget tracking
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        # Phase 1: Coarse-to-fine 1D grid search along each axis
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minimum at each x_i = 0 (in local coords), so we search for the
        # basin by finding the best 1D slice for each coordinate.
        
        dim = self.dim
        
        # Start with a coarse grid
        # We'll do a multi-resolution search
        # Resolution levels: from coarse to fine
        # For each level, we evaluate a grid along each axis
        
        # Strategy: 
        # 1. Coarse grid: 21 points per axis -> 21*5 = 105 evaluations
        # 2. Medium grid: 41 points per axis in a window -> 41*5 = 205 evaluations
        # 3. Fine grid: 81 points per axis in a tighter window -> 81*5 = 405 evaluations
        # Total: ~715, too many. Let's be smarter.
        
        # Better: iterative refinement
        # Level 1: 11 points per axis (11*5=55 calls)
        # Level 2: 21 points in window around best (21*5=105 calls)
        # Level 3: 41 points in tighter window (41*5=205 calls)
        # Total: 365 calls for grid search, leaving ~135 for Nelder-Mead
        
        # Actually, let's use a different approach:
        # Evaluate at a set of candidate points that cover the space,
        # then refine around the best.
        
        # Phase 1a: Very coarse - evaluate at key points
        # The Rastrigin function has local minima at every integer.
        # The global minimum is at the integer lattice point closest to the true minimum.
        # We need to find which integer lattice point is the global basin.
        
        # Coarse grid: 5 points per axis = 5^5 = 3125, too many.
        # Let's use a separable approach.
        
        # For each axis i, find the best value of x_i by scanning,
        # holding other axes at their current best.
        
        # Initialize with center
        x = np.zeros(dim)
        fx = call_func(x)
        
        # Coarse 1D scan for each axis
        # 9 points per axis
        coarse_points = np.linspace(-5, 5, 9)
        
        for axis in range(dim):
            best_val = fx
            best_pt = x[axis]
            for pt in coarse_points:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val
        
        # Now we have a rough estimate. Refine with medium grid.
        # 17 points in a window of width 2 around current best for each axis
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 1.0), min(5, center + 1.0), 17)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val
        
        # Check time
        if time.time() > deadline:
            return x, fx
        
        # Fine grid: 25 points in a window of width 0.5
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 0.5), min(5, center + 0.5), 25)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val
        
        if time.time() > deadline:
            return x, fx
        
        # Phase 2: Nelder-Mead polish
        # Use scipy's minimize with Nelder-Mead
        from scipy.optimize import minimize
        
        best_x = x.copy()
        best_f = fx
        
        # Nelder-Mead from current best
        try:
            result = minimize(
                call_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            return best_x, best_f
        
        # If we still have budget, do a second Nelder-Mead with tighter tolerance
        if calls_made < self.budget - 10:
            try:
                result = minimize(
                    call_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
