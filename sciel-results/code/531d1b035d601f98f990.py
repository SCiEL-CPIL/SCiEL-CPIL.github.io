"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start global-to-local hybrid optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin to avoid timeout (300s limit, 280s budget)
        deadline = start_time + 280.0

        # 1. Global Phase: Coarse grid search to find the basin of attraction
        # The function is rotated and shifted. A grid of 3^5=243 points is cheap
        # and ensures we likely find a point closer to the global optimum than a single random start.
        
        # We allocate 243 calls to the grid.
        # Grid points: -5, 0, 5 in each dimension.
        # This covers the box [-5, 5]^dim.
        # However, the optimum is in [-4, 4]^dim.
        # Let's use points: -4, 0, 4. This is tighter and more likely to hit the basin.
        # Actually, -5, 0, 5 is safer if the shift is large, but -4,0,4 is better for resolution.
        # Let's use 5 points per dim? 5^5 = 3125. Too many.
        # Let's stick to 3^5 = 243.
        # Points: -4, 0, 4.
        
        grid_vals = np.array([-4.0, 0.0, 4.0])
        
        # Generate grid
        # Use meshgrid for simplicity
        grids = np.meshgrid(*([grid_vals] * dim), indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate grid
        # We can do this in a loop to check time and budget
        for p in grid_points:
            if time.time() > deadline or budget <= 0:
                break
            fp = func(p)
            budget -= 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
                
        if best_x is None:
            # Fallback if grid failed or no budget
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        # 2. Local Phase: (1+1)-ES refinement
        # Start from the best grid point.
        x = best_x
        fx = best_f
        
        # Adaptive step size
        # Initial sigma should be small enough to stay in the basin but large enough to move.
        # Grid spacing is 4.0. Let's start with sigma = 1.0.
        sigma = 1.0
        
        # Success rule parameters
        # Standard (1+1)-ES uses 1/5 success rule.
        # Here we use a simple adaptation: increase on success, decrease on failure.
        # To be robust, we use a time-based loop.
        
        while budget > 0:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * np.random.randn(dim), -5.0, 5.0)
            fy = func(y)
            budget -= 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                
                # If sigma becomes too small, we might be stuck.
                # Weierstrass is multi-modal. If we get stuck, we could restart locally.
                # But with only 500 budget, and 243 used, we have ~250 left.
                # A simple restart strategy: if no improvement in last 50 steps, restart from a random point near x?
                # Or just keep going. The 1/5 rule is usually sufficient for smooth-ish functions.
                # Weierstrass is fractal, so "smooth" is tricky.
                # Let's add a simple restart: if sigma < 0.01, pick a new random point in [-1,1] around x?
                # No, let's just let it run. The grid found a good basin.
                
        return x, fx
# EVOLVE-BLOCK-END
