"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_calls = 0
        best_x = None
        best_f = float('inf')

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal total_calls
            if total_calls >= self.budget:
                # Raise an exception to stop the optimizer immediately
                # We catch this in the caller
                raise ValueError("Budget exceeded")
            total_calls += 1
            # Ensure x is within bounds for safety, though L-BFGS-B handles bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            return val

        # Generate starting points
        # 1. Center
        # 2. Corners (scaled to be inside)
        # 3. Random points
        rng = np.random.default_rng(42)
        start_points = []
        start_points.append(np.zeros(n))
        
        # Add some structured points
        for i in range(4):
            pt = rng.uniform(-4, 4, n)
            start_points.append(pt)
            
        # Add corners
        for _ in range(4):
            pt = rng.choice([-4, 4], n)
            start_points.append(pt)

        # Shuffle to ensure diversity in early calls? 
        # Actually, for AOCC, we want the best result ASAP.
        # Center is often a good bet for symmetric functions.
        # Let's try Center first, then random.
        
        bounds = [(-5.0, 5.0)] * n

        for start_pt in start_points:
            if total_calls >= self.budget:
                break
            
            # Allocate remaining budget to this run
            # L-BFGS-B might converge quickly.
            # We don't know exactly how many calls it will use, so we just let it run
            # until it converges or budget is hit.
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except ValueError:
                # Budget exceeded during this run
                break
            except Exception:
                # Other errors, skip this start point
                continue

            # If we haven't used the full budget, we can try more starts
            # But if we converged, the next start might find a better global min.

        # If no optimization was successful (e.g. exception before first call in wrapped_func? 
        # No, wrapped_func increments total_calls. 
        # If total_calls is 0, it means an exception happened before any func call in minimize?
        # minimize calls func at the start. So total_calls >= 1 if minimize started.
        
        if best_x is None:
            # Fallback: just return center if nothing worked
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
