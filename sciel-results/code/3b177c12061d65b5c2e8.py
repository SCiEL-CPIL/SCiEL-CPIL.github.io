"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start population-based optimizer for BBOB f14.
    Strategy:
    1. Spend 20% of budget on a coarse grid of candidate starting points.
    2. Select the top 3 candidates.
    3. Spend remaining 80% on fast local refinement (fixed step random walk)
       for each candidate, allocating time proportionally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def refine(x_start, fx_start, max_calls):
            nonlocal calls_used
            # Local refinement with fixed step size
            # Start with a moderate step and shrink it as we get closer
            # Since we don't know the scale, we use a heuristic:
            # Step size proportional to remaining budget? No, just a fixed schedule.
            # For f14, the minimum is usually not too far from the grid points if grid is good.
            # Let's use a simple adaptive step: start at 0.5, shrink by 0.95 each step if success.
            
            x = x_start.copy()
            fx = fx_start
            sigma = 0.5
            
            for _ in range(max_calls):
                if check_time():
                    break
                
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    # Slightly increase or keep sigma if we are moving well?
                    # Actually, shrinking is safer to converge.
                    sigma *= 0.98
                else:
                    sigma *= 0.95
                
                # Prevent sigma from becoming too small
                if sigma < 1e-5:
                    sigma = 1e-5

            return x, fx

        # Phase 1: Coarse Grid Exploration
        # Allocate 20% of budget to exploration
        explore_budget = int(budget * 0.2)
        
        # Create a grid of starting points
        # In 5D, a full grid is expensive. Use a sparse grid.
        # Points: -4, -2, 0, 2, 4 -> 5 points per dim -> 5^5 = 3125 points.
        # 3125 is too many for 20% of 500 (100 calls).
        # Let's use 3 points per dim: -3, 0, 3 -> 3^5 = 243 points.
        # 243 > 100. 
        # Let's use a random subset of the grid or a smaller grid.
        # 2 points per dim: -3, 3 -> 2^5 = 32 points. This fits well.
        # Or mix: some random points in [-4, 4].
        
        candidates = []
        
        # Fixed grid points: -3, 0, 3
        grid_vals = [-3.0, 0.0, 3.0]
        # Generate all combinations
        import itertools
        grid_points = list(itertools.product(grid_vals, repeat=dim))
        
        # Shuffle to avoid bias if budget runs out
        rng.shuffle(grid_points)
        
        # Evaluate grid points until explore_budget is used or grid is exhausted
        for point in grid_points:
            if calls_used >= explore_budget or check_time():
                break
            x = np.array(point)
            fx = func(x)
            calls_used += 1
            candidates.append((fx, x))
            
        # If we still have budget and grid is small, add random points
        # But 243 points is likely enough if we have 100 budget. 
        # Wait, 100 budget means we only evaluate 100 points.
        # Let's sort candidates by fx
        candidates.sort(key=lambda t: t[0])
        
        # Select top 3 candidates for refinement
        # If fewer than 3, take all.
        top_candidates = candidates[:3]
        
        if not top_candidates:
            # Fallback: start at origin
            x = np.zeros(dim)
            fx = func(x)
            calls_used += 1
            top_candidates = [(fx, x)]
            
        # Phase 2: Local Refinement
        # Allocate remaining budget to top candidates
        remaining_budget = budget - calls_used
        
        # Distribute budget: give more to better candidates?
        # Simple: split equally among top 3.
        n_refine = len(top_candidates)
        if n_refine == 0:
            return np.zeros(dim), float('inf')
            
        # Ensure at least 1 call per candidate if possible
        calls_per_refine = max(1, remaining_budget // n_refine)
        
        best_x = None
        best_fx = float('inf')
        
        for i, (fx_init, x_init) in enumerate(top_candidates):
            # Adjust calls for the last one to use up remaining budget
            if i == n_refine - 1:
                calls_this = remaining_budget - calls_per_refine * (n_refine - 1)
            else:
                calls_this = calls_per_refine
            
            if calls_this <= 0:
                continue
                
            x_res, fx_res = refine(x_init, fx_init, calls_this)
            
            if fx_res < best_fx:
                best_fx = fx_res
                best_x = x_res
                
            if check_time():
                break
                
        if best_x is None:
            best_x = top_candidates[0][1]
            best_fx = top_candidates[0][0]
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
