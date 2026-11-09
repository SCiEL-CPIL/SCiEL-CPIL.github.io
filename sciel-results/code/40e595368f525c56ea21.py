"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for Schwefel-style f20.
    Uses a deadline guard to stay within wall-clock limits.
    Strategy:
      1. Perform a coarse grid search to find a promising basin.
      2. Refine the best point using Nelder-Mead (scipy.optimize.minimize).
      3. If time remains, attempt further refinement or restart from the best point.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        max_calls = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, calls
            if calls >= max_calls:
                return best_f
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return float('inf')

        # Phase 1: Coarse Grid Search
        # Evaluate a small grid to find a good starting point for local search
        # 5 dimensions, 3 points per dimension = 3^5 = 243 points.
        # This is well within the budget of 500.
        grid_points = []
        # Use 3 points per dimension: -4, 0, 4 (covering most of [-5,5])
        levels = [-4.0, 0.0, 4.0]
        
        # Generate grid
        import itertools
        for combo in itertools.product(levels, repeat=self.dim):
            grid_points.append(np.array(combo))
        
        # Shuffle to avoid bias if the function has a weird structure
        rng.shuffle(grid_points)
        
        for x in grid_points:
            if time.time() > deadline:
                break
            if calls >= max_calls:
                break
            safe_call(x)
            
        # If we didn't find any valid point (shouldn't happen), fallback
        if best_x is None:
            x_init = rng.uniform(-5, 5, self.dim)
            safe_call(x_init)
            if best_x is None:
                return np.zeros(self.dim), float('inf')

        # Phase 2: Local Refinement with Nelder-Mead
        # Use the best point from the grid as the starting point
        x0 = best_x.copy()
        
        # Nelder-Mead options
        # maxiter controls the number of iterations, not function calls directly,
        # but we wrap the function to count calls and enforce limits.
        
        # We need to be careful with scipy's minimize as it might make many calls.
        # We will use a wrapper that respects the call budget and deadline.
        
        def wrapped_func(x):
            if time.time() > deadline:
                return best_f
            if calls >= max_calls:
                return best_f
            return safe_call(x)

        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # Phase 3: Further Refinement if time and budget remain
        if time.time() < deadline and calls < max_calls:
            # Try a few more Nelder-Mead runs from slightly perturbed best_x
            for _ in range(3):
                if time.time() > deadline or calls >= max_calls:
                    break
                x_start = best_x + rng.normal(0, 0.1, self.dim)
                x_start = np.clip(x_start, -5, 5)
                try:
                    res = minimize(wrapped_func, x_start, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
