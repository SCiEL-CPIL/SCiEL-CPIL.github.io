"""Multi-Resolution Coordinate Descent with Adaptive Step Halving for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Resolution Coordinate Descent (MRCD) with Adaptive Step Halving.
    
    Strategy:
    1. Sparse Global Search: Evaluate a coarse grid to identify promising regions (basins).
    2. Local Deterministic Refinement: For each promising region, perform coordinate descent 
       with adaptive step sizes. 
    3. Plateau Handling: If a step does not improve the value (indicating we are on a plateau 
       or have crossed a step boundary without entering a better region), halve the step size 
       to refine the position within the plateau or find the exact boundary. If the step 
       improves, keep the step size or grow it slightly if we are far from the optimum, 
       but primarily rely on halving to converge to the local minimum of the step function.
    4. Multi-Start: Run this local refinement from multiple initial points found in the 
       global search phase to avoid getting stuck in sub-optimal local minima of the 
       ellipsoidal steps.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Random number generator for tie-breaking or slight perturbations if needed
        rng = np.random.default_rng(42)
        
        def check_time():
            return time.time() > deadline
            
        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            if check_time():
                return None
            try:
                f_val = func(x)
                calls_used += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Sparse Global Search
        # We want to cover the domain [-5, 5]^d.
        # A grid of 3 points per dimension gives 3^5 = 243 points for d=5.
        # This is a significant chunk of the 500 budget but ensures we find at least one good basin.
        # Let's use a slightly sparser grid to leave room for refinement.
        # Grid points: -5, 0, 5 is too coarse. 
        # Let's use -4, 0, 4 to stay away from boundaries initially, or -5, 0, 5.
        # The minimizer is in [-4, 4].
        
        # Let's allocate 25% of budget to global search.
        # 125 calls for global search.
        # 3^5 = 243 is too many. 
        # Let's use a Latin Hypercube or random sampling for the global phase to be more efficient,
        # but the directive asks for "sparse grid". 
        # A grid of 2 points per dim? 2^5 = 32 points. Too sparse.
        # Let's do a hybrid: A coarse grid of 3^d points is standard. 
        # For d=5, 3^5 = 243. This leaves 257 calls for refinement.
        # If we use 5 starts for refinement, we have ~50 calls per start.
        # This might be tight. 
        # Let's reduce the grid density. 
        # Use 2 points per dimension: -4, 4? No, that misses the center.
        # Use 3 points: -4, 0, 4. 243 points.
        
        # Alternative: Use a smaller set of "seeds" for local search.
        # Let's try 10 random points + the center + the 8 corners of [-4,4]^5? 
        # No, let's stick to the directive: "initialize with a sparse grid".
        
        # Let's use a grid of size 3 for d<=4, and for d=5, maybe 3x3x3x2x2? 
        # Actually, 3^5 = 243 is acceptable if the local search is very fast (deterministic).
        # Coordinate descent is very cheap in computation, only function calls matter.
        
        grid_points = []
        # Generate grid points in [-4, 4] to be safe
        # 3 points per dimension: -4, 0, 4
        coords = [-4.0, 0.0, 4.0]
        
        # For d=5, 3^5 = 243 points.
        # Let's evaluate them.
        # To save time/calls, we can be smarter. 
        # But let's try the full grid first. If it's too expensive, we can prune.
        
        # Generate all combinations
        import itertools
        grid_coords = list(itertools.product(coords, repeat=d))
        
        # Shuffle to avoid bias in early stopping? 
        # Actually, for AOCC, finding a good point early is good.
        # Center is usually a good start. 
        # Let's sort by distance to center? No, random order is fine, but center first is strategic.
        
        # Ensure center is first
        center = np.zeros(d)
        grid_list = []
        for gp in grid_coords:
            gp_arr = np.array(gp)
            if np.allclose(gp_arr, center):
                grid_list.insert(0, gp_arr)
            else:
                grid_list.append(gp_arr)
        
        # Evaluate grid points
        # We have budget 500. 
        # If we use 243 for grid, we have 257 left.
        # Let's limit grid evaluation to 100 points to be safer?
        # 3^5 = 243. Let's just evaluate all 243. It's deterministic and fast.
        
        # However, if we run out of budget, we return.
        # Let's prioritize the center and then random others?
        # No, the grid is systematic.
        
        # To optimize AOCC, we want to find a low value quickly.
        # The center is a good bet. 
        # Let's evaluate the center first.
        
        # Re-order: Center, then corners of the grid, then rest?
        # Let's just evaluate in the order generated, but put center first.
        
        # Actually, let's reduce the grid size to leave more budget for refinement.
        # Use 2 points per dim? No.
        # Let's use a random sample of size 50 from the grid? 
        # No, "sparse grid" implies structure.
        
        # Let's use a grid of 3 points for the first 3 dims and 2 for the last 2?
        # 3*3*3*2*2 = 108 points. This is better.
        
        # Let's implement a recursive grid generator with varying resolution.
        # Or simpler: Just use 3^d but if d=5, it's 243. 
        # Let's stick to 243. It's less than half the budget.
        
        # Evaluate grid
        grid_best_x = None
        grid_best_f = np.inf
        
        for gp in grid_list:
            if calls_used >= budget or check_time():
                break
            f_val = safe_call(gp)
            if f_val is not None:
                if f_val < grid_best_f:
                    grid_best_f = f_val
                    grid_best_x = gp.copy()
        
        if calls_used >= budget or check_time():
            return best_x, best_f

        # Phase 2: Local Refinement (Multi-Resolution Coordinate Descent)
        # We have several candidate starts. 
        # We will pick the top K candidates from the grid to refine.
        # Since we didn't store all grid results, we only have the global best so far.
        # To be more robust, we should store the top few grid points.
        
        # Let's redo the grid evaluation to store top K.
        # Reset calls? No, we already used them. 
        # This is a problem. We need to plan the budget.
        
        # Revised Plan:
        # 1. Evaluate 50 random points + center. (51 calls)
        # 2. Select the best 5 points from these.
        # 3. Run Coordinate Descent from these 5 points, using the remaining ~449 calls.
        #    449 / 5 = ~90 calls per local search.
        
        # This is a more balanced approach for "Multi-Start".
        # The directive says "initialize with a sparse grid". 
        # 50 random points is not a grid. 
        # Let's use a grid of 3^4 * 2 = 162 points? 
        # Or 3^5 = 243.
        
        # Let's stick to the 243 grid idea but only keep the best 10.
        # I will re-implement the grid loop to store results.
        
        # Wait, I already executed the grid loop above in the code flow? 
        # No, the code above was just a sketch. I need to write the actual code.
        # I will restart the logic in the __call__ method properly.
        
        # --- RESTART LOGIC IN ACTUAL CODE ---
        
        # The code above was just for my thinking. 
        # I will now write the final code block.
        
        pass # Placeholder, real code follows in the block below.
        
# The above was a thought process. Here is the actual implementation.

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        rng = np.random.default_rng(42)
        
        def check_time():
            return time.time() > deadline
            
        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            if check_time():
                return None
            try:
                f_val = func(x)
                calls_used += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Sparse Grid Initialization
        # Use a grid of 3 points per dimension: -4, 0, 4
        # For d=5, this is 243 points.
        # We will evaluate these and keep the top 10 for local search.
        
        coords = [-4.0, 0.0, 4.0]
        import itertools
        grid_coords = list(itertools.product(coords, repeat=d))
        
        # Sort by distance to center to prioritize good starts
        center = np.zeros(d)
        def dist_to_center(p):
            return np.linalg.norm(np.array(p) - center)
        
        grid_coords.sort(key=dist_to_center)
        
        top_grid_points = []
        top_grid_fvals = []
        
        for gp in grid_coords:
            if calls_used >= budget or check_time():
                break
            gp_arr = np.array(gp)
            f_val = safe_call(gp_arr)
            if f_val is not None:
                top_grid_points.append(gp_arr)
                top_grid_fvals.append(f_val)
                
                # Keep only top 10 based on f_val
                if len(top_grid_points) > 10:
                    # Find index of max f_val
                    max_idx = np.argmax(top_grid_fvals)
                    top_grid_points.pop(max_idx)
                    top_grid_fvals.pop(max_idx)
        
        if calls_used >= budget or check_time():
            return best_x, best_f

        # If we didn't find any points (shouldn't happen), return
        if not top_grid_points:
            return best_x, best_f

        # Phase 2: Multi-Resolution Coordinate Descent
        # For each of the top 10 points, run a local coordinate descent.
        # Budget allocation:
        # Remaining budget = budget - calls_used
        # Distribute evenly among the 10 starts.
        
        n_starts = len(top_grid_points)
        remaining_budget = budget - calls_used
        budget_per_start = max(1, remaining_budget // n_starts)
        
        for i in range(n_starts):
            if calls_used >= budget or check_time():
                break
            
            x = top_grid_points[i].copy()
            f_curr = top_grid_fvals[i]
            
            # If this point is worse than global best, we might still want to search 
            # in case it leads to a better basin, but we can limit its budget.
            # For simplicity, give equal budget.
            
            # Adaptive Step Halving Coordinate Descent
            # Initial step size: 2.0 (covers half the domain roughly)
            step_size = 2.0
            min_step = 1e-4
            max_iter = budget_per_start
            
            # To avoid wasting calls if we are stuck, we can break early if no improvement 
            # for a certain number of steps? 
            # But halving will eventually make steps tiny.
            # We'll just run until budget for this start is exhausted or time runs out.
            
            for _ in range(max_iter):
                if calls_used >= budget or check_time():
                    break
                if step_size < min_step:
                    break
                    
                improved = False
                for j in range(d):
                    if calls_used >= budget or check_time():
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[j] += step_size
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    
                    f_trial = safe_call(x_trial)
                    if f_trial is None:
                        break
                    if f_trial < f_curr:
                        x = x_trial
                        f_curr = f_trial
                        improved = True
                        # Do not halve step immediately, keep it to explore further in this direction?
                        # Or halve to refine? 
                        # Standard CD often keeps step if improved, or reduces if not.
                        # For step functions, if we improved, we might be on a new plateau.
                        # Let's halve the step to refine the position within the new region.
                        # This ensures we don't overshoot the next step.
                        step_size *= 0.5
                        # Restart coordinate loop? Or continue?
                        # If we continue, we might try other axes.
                        # Let's break the axis loop and restart with the new step size?
                        # Or just continue.
                        # Continuing is fine.
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[j] -= step_size
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            break
                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            step_size *= 0.5
                        else:
                            # No improvement in either direction for this axis.
                            # This suggests we are at a local minimum for this axis 
                            # or on a flat plateau where the step is too large to detect change?
                            # If on a plateau, a smaller step might reveal a gradient? 
                            # No, step functions are flat. 
                            # If no improvement, halve the step to check for finer structure.
                            step_size *= 0.5
                            
                    # If step_size is very small, we can skip further halving for this axis?
                    # The loop condition handles it.
                
                # If we completed all axes without any improvement, and step size is large,
                # we might be stuck. Halving happens inside the loop.
                # If step_size becomes < min_step, we break.
                
                # Optimization: If no improvement in a full pass, and step_size is already small, 
                # we can break early to save budget for other starts.
                # But we don't know if "no improvement" was due to plateau or true min.
                # Halving is the safe bet.
                
        return best_x, best_f

# EVOLVE-BLOCK-END
