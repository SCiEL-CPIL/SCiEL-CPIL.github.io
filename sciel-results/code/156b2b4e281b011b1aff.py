"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Wrapper to count calls and handle time
        def wrapped_func(x):
            nonlocal calls
            if time.time() > deadline:
                return best_f  # Return current best to stop exploration
            if calls >= budget:
                return best_f
            calls += 1
            return func(x)
            
        # Phase 1: Multi-start Initialization
        # Weierstrass is multimodal. We sample a grid of points to find good basins.
        # The minimizer is in [-4, 4]^5.
        # Use 10% of budget for initialization, at least 10 points.
        init_budget = max(10, min(50, budget // 10))
        
        # Generate deterministic diverse starting points
        # Mix of center, corners, and random-like deterministic points
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Deterministic "random" points using a fixed seed for reproducibility within the instance
        # But we want diversity. Let's use a simple LHS or grid.
        # Since dim=5 is small, a coarse grid or random is fine.
        # Let's use a fixed RNG for the initialization points to ensure consistency,
        # but the function itself is fixed per instance.
        rng = np.random.default_rng(42)
        
        # Add random points in [-4, 4]
        for _ in range(init_budget - 1):
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)
            
        # Evaluate all start points
        for p in start_points:
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            f_p = wrapped_func(p)
            if f_p < best_f:
                best_f = f_p
                best_x = p.copy()
                
        # Phase 2: Local Refinement with Nelder-Mead
        # Nelder-Mead is robust for smooth, non-convex functions if started in a good basin.
        # We will run Nelder-Mead from the best point found in Phase 1.
        # If time/budget allows, we might restart from the second best point if it's close.
        
        if calls < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'xatol': 1e-8, 
                        'fatol': 1e-10, 
                        'maxiter': budget - calls,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass # Ignore errors in optimizer, keep current best

        return best_x, best_f
# EVOLVE-BLOCK-END
