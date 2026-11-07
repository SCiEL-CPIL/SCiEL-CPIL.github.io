"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent with adaptive axis prioritization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget
        calls_made = 0

        global_best_x = np.zeros(dim)
        global_best_f = np.inf

        # Initial evaluation at center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        global_best_x = x0.copy()
        global_best_f = f0

        # Adaptive coordinate descent
        # Start with a large step size and refine
        # We use a list of step sizes that decrease geometrically
        # But we adapt the number of points based on budget
        # Strategy:
        # 1. Coarse grid search to find a good basin
        # 2. Fine coordinate descent to refine
        
        # Phase 1: Coarse Search
        # We will evaluate points on a grid around the current best
        # Step sizes: 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125...
        # For each step size, we try moving +/- step in each dimension
        # We prioritize dimensions that have yielded the most improvement recently
        
        # To keep it deterministic and simple, we iterate through scales
        # and for each scale, we do a coordinate descent pass
        
        # Let's define a set of scales
        # We have 500 calls. 1 used. 499 left.
        # If we do 10 scales, and for each scale we try 2*dim = 10 points (if we only go +/-)
        # That's 100 points.
        # We can afford more.
        
        # Better strategy: 
        # Use a "zoom" approach.
        # Start at center.
        # For scale in [4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        #    For each dimension d:
        #       Try x[d] +/- scale
        #       If better, update x
        #       (This is standard coordinate descent, but we can enhance it)
        
        # Enhanced: Multi-start at coarse scales?
        # No, deterministic. 
        # Let's stick to a robust coordinate descent with multiple scales.
        # To avoid getting stuck in a local minimum early, we can do a "broad" search first.
        
        # Broad Search:
        # Evaluate a grid of points.
        # With dim=5, a 3^5 grid is 243 points. 
        # Points at -4, 0, 4 for each dim.
        # This covers [-4, 4] reasonably well.
        # 243 calls.
        # Then refine the best point with coordinate descent using remaining budget.
        
        # Let's try this:
        # 1. Grid search on {-4, 0, 4}^5 -> 243 calls.
        # 2. Find best point.
        # 3. Coordinate descent with decreasing steps from 2.0 down to 1e-4.
        
        # Wait, 243 calls is nearly half the budget.
        # Is it efficient?
        # AOCC rewards early progress.
        # Grid search might find a good point quickly if the global min is in a coarse cell.
        # But if the min is at (1, 1, 1, 1, 1), grid search at 0, 4, -4 might miss it.
        
        # Alternative: 
        # Start at 0.
        # Use a set of step sizes: [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        # For each step size, do a full coordinate descent pass (try +/- in each dim).
        # This is 12 scales * 2 * 5 = 120 calls.
        # We have ~380 calls left.
        # We can repeat the process or use smaller steps.
        
        # Let's implement a flexible loop:
        # current_x = zeros
        # current_f = func(current_x)
        
        # Scales: Start large, go small.
        # If we find no improvement in a scale, we can skip to the next or restart?
        # Deterministic: just go to next smaller scale.
        
        # To make it more robust, we can do "random" deterministic restarts?
        # No, "deterministic multi-scale coordinate descent" is the directive.
        
        # Let's try a hybrid:
        # 1. Coarse grid: 3 points per dim at {-3, 0, 3}. 3^5 = 243 points.
        #    This is expensive but covers the box [-3, 3] which contains the min.
        #    Actually, min is in [-4, 4].
        #    Let's use {-4, -2, 0, 2, 4}? 5^5 = 3125 points. Too many.
        #    {-3, 0, 3} is 243.
        #    If we spend 243 calls, we have 257 left for refinement.
        #    Refinement: 10 scales * 10 points = 100 calls.
        #    Total 343 calls.
        
        # Does grid search give early progress?
        # The first point evaluated is (-3, -3, -3, -3, -3).
        # If the min is near (0,0,0,0,0), this is far.
        # AOCC will be low for the first 200 calls.
        # Coordinate descent from 0 might be better for AOCC if min is near 0.
        # But min is "hidden seeded shift". It could be anywhere in [-4, 4].
        
        # Compromise:
        # Start at 0.
        # Do coordinate descent with large steps first.
        # Steps: 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001
        # Each step: 10 evaluations (2 per dim).
        # 15 steps * 10 = 150 evaluations.
        # This is fast.
        # We have 350 evaluations left.
        # We can repeat the cycle with even smaller steps or do a local search.
        
        # Let's implement:
        # 1. Initialize at 0.
        # 2. Loop through a list of step sizes.
        # 3. For each step size, do coordinate descent (try +/- in each dim).
        # 4. If no improvement in a full pass, continue to next smaller step.
        # 5. If budget remains, repeat with even smaller steps.
        
        # To handle the "hidden shift", starting at 0 might be bad if min is at (4, 4, 4, 4, 4).
        # But with step 4.0, we try (4, 0, 0, 0, 0), (-4, 0...), etc.
        # In one pass of step 4.0, we cover the boundaries.
        # If min is at (4, 4, 4, 4, 4), we will find (4, 0, 0, 0, 0) then (4, 4, 0, 0, 0) etc.
        # It will take several passes to move all dimensions.
        # Pass 1 (step 4.0):
        #   Dim 0: try 4, -4. Suppose 4 is better. x = [4, 0, 0, 0, 0].
        #   Dim 1: try 4, -4. Suppose 4 is better. x = [4, 4, 0, 0, 0].
        #   ...
        #   End of pass 1: x = [4, 4, 4, 4, 4].
        # So one pass of step 4.0 can potentially jump to the corner!
        # This is very efficient for finding the basin.
        
        # So, starting at 0 and doing coordinate descent with large steps is a strong strategy.
        
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
        
        x = np.zeros(dim)
        fx = global_best_f
        
        for scale in step_sizes:
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break
                
            improved = False
            # Coordinate descent pass
            for d in range(dim):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                    
                # Try positive
                x_new = x.copy()
                x_new[d] += scale
                if x_new[d] > 5.0: x_new[d] = 5.0
                if x_new[d] < -5.0: x_new[d] = -5.0
                
                # Only evaluate if it's different from current x
                if not np.allclose(x_new, x):
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved = True
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()
                
                # Try negative
                x_new = x.copy()
                x_new[d] -= scale
                if x_new[d] > 5.0: x_new[d] = 5.0
                if x_new[d] < -5.0: x_new[d] = -5.0
                
                if not np.allclose(x_new, x):
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        improved = True
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()
                            
            # If no improvement in this scale, we could break early?
            # No, smaller scales might still help if we are near a minimum but not exactly at it.
            # But if we are stuck, smaller scales won't help unless we are very close.
            # Let's continue.
            
        # If we have budget left, do a final fine-tuning or restart?
        # Since it's deterministic, we can just add more tiny steps.
        extra_steps = [0.000005, 0.000002, 0.000001]
        for scale in extra_steps:
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 1.0:
                break
            
            for d in range(dim):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                    
                x_new = x.copy()
                x_new[d] += scale
                x_new[d] = np.clip(x_new[d], -5.0, 5.0)
                if not np.allclose(x_new, x):
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()
                
                x_new = x.copy()
                x_new[d] -= scale
                x_new[d] = np.clip(x_new[d], -5.0, 5.0)
                if not np.allclose(x_new, x):
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
