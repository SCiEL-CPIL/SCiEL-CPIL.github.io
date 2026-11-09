"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid search with adaptive refinement for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid search
        # Divide [-5, 5]^dim into a grid. For dim=5, use 3 points per dimension initially.
        # Total points: 3^5 = 243. This fits within budget.
        # But we need to be careful about the deadline.
        
        # Start with a very coarse grid: 2 points per dim -> 2^5 = 32 points
        # Then refine around the best.
        
        # Actually, a hierarchical approach:
        # Level 0: Grid with step 5.0 (2 points per dim: -5, 0, 5? No, 3 points: -5, 0, 5)
        # Let's use 3 points per dimension for the first pass: 3^5 = 243 calls.
        # That's less than 500.
        
        # Generate grid points
        # For each dimension, points are -5, 0, 5
        grid_points = np.array(np.meshgrid(*([[-5.0, 0.0, 5.0]] * dim))).T.reshape(-1, dim)
        
        # Shuffle to get diversity early? No, deterministic is fine for reproducibility, 
        # but the function is shifted/rotated. Random order might help if we hit a good spot early.
        # However, the scoring rewards early progress. 
        # Let's evaluate in a random order to potentially find a good region faster.
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        
        for idx in perm:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = grid_points[idx]
            eval_point(x)
        
        # Phase 2: Refine around the best point found
        # Use a local grid search centered at best_x
        # Step size starts at 2.0 (half of the coarse step 5.0? No, coarse step was 5.0 between -5,0,5)
        # Let's use a step size that covers the neighborhood.
        
        # We have remaining budget.
        # Strategy: Iteratively refine.
        # Step size: start with 2.0, then 1.0, 0.5, 0.25, ...
        # At each step, evaluate a grid of points around best_x.
        # Grid: best_x + {-step, 0, step} in each dimension.
        # This is 3^dim points per step.
        # For dim=5, 3^5 = 243 points.
        # We can afford 1-2 refinement steps if we used 243 in Phase 1.
        # 500 - 243 = 257. So we can do one more 3^5 grid.
        
        # But maybe Phase 1 should be coarser?
        # If we use 2 points per dim in Phase 1: 2^5 = 32 points.
        # Then we have 468 points for refinement.
        # Refinement 1: Step 2.5, 3^5=243 points. Total 275.
        # Refinement 2: Step 1.25, 3^5=243 points. Total 518 > 500.
        # So we can do Phase 1 (32) + Refine 1 (243) + part of Refine 2.
        
        # Let's restart the logic with a better allocation.
        # Reset state? No, we already used calls.
        # Let's just continue refining from the current best.
        
        # Current step size for refinement
        step = 2.0
        while calls < budget and step > 1e-6:
            if time.time() > deadline:
                break
            
            # Generate grid points around best_x
            # Points: best_x[i] + offset, where offset in {-step, 0, step}
            # To avoid re-evaluating the center (already best), we can skip it or include it.
            # Including it is fine, it just wastes 1 call.
            
            offsets = [-step, 0.0, step]
            grid_points = np.array(np.meshgrid(*([offsets] * dim))).T.reshape(-1, dim)
            grid_points = best_x + grid_points
            
            # Clip to bounds
            grid_points = np.clip(grid_points, -5.0, 5.0)
            
            # Shuffle
            perm = rng.permutation(len(grid_points))
            
            for idx in perm:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                x = grid_points[idx]
                eval_point(x)
            
            # Reduce step size
            step *= 0.5
            
        # Phase 3: If we still have budget, do a final local search with very small steps
        # Or just continue the loop above. The loop above handles it.
        
        # Return best
        return best_x, best_f

# EVOLVE-BLOCK-END
