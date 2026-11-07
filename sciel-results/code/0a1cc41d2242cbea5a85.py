"""Combined Grid Search and L-BFGS-B for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: Coarse grid search followed by L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Phase 1: Coarse Grid Search
        # We use a grid of 3 points per dimension: -5, 0, 5
        # This gives 3^5 = 243 evaluations.
        # If budget is small, we might need to reduce this, but budget=500 is fixed.
        # 243 is < 500, so it's safe.
        
        # Generate grid points
        # To be more robust against shifts, maybe use [-5, 0, 5] is too coarse if shift is large?
        # The minimizer is in [-4, 4]. [-5, 0, 5] covers it.
        # Let's use a slightly finer grid if possible? 4^5 = 1024 > 500. So 3 is max for full grid.
        # Alternatively, use random points for Phase 1 to avoid grid bias?
        # The directive says "coarse grid search". I will stick to grid.
        
        grid_vals = np.array([-5.0, 0.0, 5.0])
        # Create meshgrid
        grids = np.meshgrid(*([grid_vals] * dim), indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=-1)
        
        # Limit grid points to budget/2 to leave room for polish
        max_grid_points = budget // 2
        if len(grid_points) > max_grid_points:
            # If grid is too big, subsample or just take first max_grid_points
            # For dim=5, 3^5=243. 500//2 = 250. So it fits.
            grid_points = grid_points[:max_grid_points]
            
        best_x = None
        best_fx = float('inf')
        calls_made = 0

        for p in grid_points:
            if time.time() > deadline:
                break
            if calls_made >= budget - 50: # Reserve 50 for polish
                break
                
            # Clip to box (grid is already in box, but good practice)
            p_clipped = np.clip(p, -5, 5)
            f_val = func(p_clipped)
            calls_made += 1
            
            if f_val < best_fx:
                best_fx = f_val
                best_x = p_clipped.copy()

        # Phase 2: L-BFGS-B Polish
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None and time.time() < deadline:
            # L-BFGS-B is a quasi-Newton method suitable for bounded optimization
            # It uses numerical gradients by default.
            # Each iteration might cost ~dim+1 calls for gradient estimation.
            # We limit iterations to ensure we don't exceed budget.
            
            # We need a wrapper to count calls and stop if exceeded
            polish_calls = [0]
            
            def polish_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if polish_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                polish_calls[0] += 1
                return func(np.clip(z, -5, 5))
            
            try:
                # bounds for L-BFGS-B
                bounds = [(-5.0, 5.0)] * dim
                
                # maxiter is not strictly call count, but a proxy.
                # We set it high, but the wrapper stops it.
                result = minimize(
                    polish_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': remaining_budget, 
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
