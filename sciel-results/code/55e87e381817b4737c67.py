"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Global-Local Hybrid Optimizer for BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Exploration via Coarse Grid
        # Rastrigin has periodic minima. A coarse grid ensures we find a local minimum
        # early, which is crucial for AOCC (early progress counts).
        # Grid points: 3 per dimension -> 3^5 = 243 points.
        # This uses ~50% of the budget to guarantee a decent initial best.
        
        grid_points = 3
        total_grid_points = grid_points ** dim
        
        # We might not have enough budget for the full grid if dim was larger, 
        # but for dim=5, 243 is safe. 
        # We also add a few random points to catch off-grid minima or rotation effects.
        
        # Generate grid points
        # Range [-4, 4] covers the likely minimizer region, but let's use [-5, 5] 
        # to be safe with the rotation, though the problem states minimizer in [-4,4].
        # Actually, using [-4, 4] is better to avoid clipping waste if the rotation 
        # pushes things out, but the function clips. Let's stick to the known box.
        
        grid_coords = np.linspace(-4.0, 4.0, grid_points)
        
        # Generate all combinations
        grid = np.array(np.meshgrid(*([grid_coords] * dim))).T.reshape(-1, dim)
        
        # Shuffle grid to ensure we don't just evaluate in a deterministic order 
        # that might correlate with the function structure (though Rastrigin is symmetric)
        perm = rng.permutation(len(grid))
        grid = grid[perm]
        
        # Add some random points to the grid evaluation queue to increase diversity
        # Let's say 20 random points
        num_random = 20
        random_points = rng.uniform(-4.0, 4.0, size=(num_random, dim))
        
        # Interleave or just append? 
        # For AOCC, we want the *first* good point to be found ASAP.
        # Random points might find a good spot faster than a structured grid if the grid is unlucky.
        # Let's mix them: 50% grid, 50% random? Or just evaluate all grid points first?
        # Grid is more reliable for "guaranteeing" a local min.
        # Let's evaluate grid points first.
        
        search_points = np.vstack([grid, random_points])
        
        # We have budget 500. 
        # 243 grid + 20 random = 263 points.
        # Remaining budget: 500 - 263 = 237 points for local refinement.
        
        for x in search_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            evaluate(x)
            
        # If we haven't found anything (shouldn't happen), fallback to center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # Phase 2: Local Exploitation
        # Use a local search algorithm to refine the best point found.
        # Nelder-Mead is robust and requires few parameters.
        # We will run Nelder-Mead from the best point, and potentially from the top 2-3 points.
        
        # Since we don't have scipy.optimize.minimize in the "standard library" sense 
        # (though SciPy is available per contract: "Python standard library, NumPy and SciPy are available"),
        # we can use scipy.optimize.minimize.
        
        from scipy.optimize import minimize

        # Get top 3 candidates from Phase 1
        # We need to track which points were evaluated. 
        # Since we only stored the best, let's re-evaluate or keep a history?
        # Storing history of 263 points is cheap.
        
        # Let's re-implement the loop to keep history for Phase 2 selection.
        # Actually, we can just use the best point found so far. 
        # But restarting local search from multiple points is better.
        
        # Let's refine the best point found in Phase 1.
        # We have ~237 calls left.
        # Nelder-Mead with dim=5 typically needs ~100-200 evaluations for good convergence.
        
        # Strategy: Run Nelder-Mead on the global best.
        # If it converges or fails, try the next best? 
        # We didn't keep the next best. 
        # Let's modify Phase 1 to keep the top K points.
        
        # --- Revised Plan ---
        # Phase 1: Evaluate Grid + Random. Keep top 3 points.
        # Phase 2: Run Local Search (Nelder-Mead) on each of the top 3, allocating remaining budget.
        
        # Reset state for proper tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        history = [] # List of (f_val, x)

        def evaluate_track(x):
            nonlocal best_x, best_f, calls_made, history
            f_val = func(x)
            calls_made += 1
            history.append((f_val, x.copy()))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global
        for x in search_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            evaluate_track(x)
            
        # Sort history to find top candidates
        history.sort(key=lambda item: item[0])
        
        # Select top 3 distinct candidates (simple distance check to avoid duplicates)
        candidates = []
        for f_val, x in history:
            if len(candidates) >= 3:
                break
            # Check if x is too close to existing candidates
            is_unique = True
            for _, cx in candidates:
                if np.linalg.norm(x - cx) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                candidates.append((f_val, x))
        
        # Phase 2: Local Search
        # Allocate remaining budget to local search.
        # We have ~237 calls left.
        # Let's split this among the candidates.
        # Candidate 1 gets 50%, Candidate 2 gets 30%, Candidate 3 gets 20%.
        
        remaining_budget = budget - calls_made
        if remaining_budget > 0 and candidates:
            
            # Define a wrapper to count calls for local search
            local_calls = [0]
            
            def local_func(x):
                nonlocal best_x, best_f
                if local_calls[0] >= remaining_budget:
                    return best_f # Return a bad value to stop? 
                                 # Actually, minimize will stop if we raise an exception or return inf?
                                 # Better to just let it run and check time/budget in the main loop?
                                 # SciPy minimize doesn't support a callback that stops it easily without exceptions.
                                 # We'll use maxiter and options.
                                 
                local_calls[0] += 1
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val

            # We need to be careful with SciPy's minimize. 
            # It might exceed the budget.
            # We can use a custom method or just Nelder-Mead with maxiter.
            # Nelder-Mead maxiter is number of function calls? 
            # In SciPy, 'maxiter' is the maximum number of iterations. 
            # For Nelder-Mead, 1 iteration is roughly dim+1 function calls.
            
            # Let's use a simple custom local search if SciPy is risky?
            # No, SciPy is allowed and robust.
            
            # We will run Nelder-Mead on the best candidate.
            # If we have budget left, run on the second.
            
            for i, (f_val, x0) in enumerate(candidates):
                if calls_made + local_calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Allocate budget for this candidate
                if i == 0:
                    alloc = int(remaining_budget * 0.6)
                elif i == 1:
                    alloc = int(remaining_budget * 0.3)
                else:
                    alloc = int(remaining_budget * 0.1)
                
                # Reset local call counter for this specific run?
                # No, local_calls tracks total local calls. 
                # We need to ensure we don't exceed total remaining.
                
                # SciPy minimize options
                # 'maxiter' for Nelder-Mead is max number of function evaluations? 
                # Documentation: "Maximum number of iterations to perform."
                # For Nelder-Mead, it's roughly the number of simplex updates.
                # Let's be conservative.
                
                try:
                    res = minimize(
                        local_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={'maxiter': alloc, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                    )
                except Exception:
                    pass

                # Update global best if local search found something better
                # local_func already updates best_x, best_f
                
                # Check if we ran out of budget or time
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
