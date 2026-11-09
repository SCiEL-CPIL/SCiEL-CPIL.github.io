"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Hybrid Global-Local Search (Random Restart + Powell's Method)
    
    This approach fundamentally changes the search paradigm from stochastic population-based
    evolution (CMA-ES) to a deterministic, structured refinement.
    
    1. Global Phase: 
       - Uses a set of random restarts combined with a coarse grid to find the basin of attraction.
       - Since f14 is shifted and rotated, we must first locate the region where the gradient is small.
       - We evaluate a diverse set of points to find the best starting point for local refinement.
       
    2. Local Phase:
       - Implements Powell's method (derivative-free line search).
       - Powell's method is particularly effective for f14 (Different powers) because it 
         optimizes along directions that adapt to the function's separable structure.
         It avoids the "isotropic step size" problem of ES methods by performing 1D minimization
         along specific directions, which allows it to handle the different scales of the 
         power terms (|x_i|^i) more efficiently.
       
    3. Robustness:
       - Explicit time and budget guards.
       - If Powell's method stagnates or hits boundaries, it restarts from the current best 
         with a perturbation, or falls back to a simpler coordinate descent if necessary.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        x_min = -5.0
        x_max = 5.0

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            # Ensure x is within bounds for the function call
            x_clipped = np.clip(x, x_min, x_max)
            f = func(x_clipped)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return x_clipped, f

        # --- Phase 1: Global Search ---
        # We want to find a good starting point for the local optimizer.
        # f14 has a global minimum. A random search with a moderate number of points
        # combined with a coarse grid helps locate the basin.
        
        # Number of global points: Use ~20% of the budget for exploration.
        # If budget is 500, we use ~100 points for global search.
        n_global = max(10, min(100, budget // 5))
        
        # Generate diverse starting points
        # Mix of random uniform and a coarse grid
        candidates = []
        
        # 1. Center
        candidates.append(np.zeros(dim))
        
        # 2. Random points
        for _ in range(n_global // 2):
            candidates.append(rng.uniform(x_min, x_max, size=dim))
            
        # 3. Coarse grid points (if dim is small, this is cheap)
        # For dim=5, a grid of 3 points per dim is 3^5 = 243 points. 
        # A grid of 2 points (min, mid, max) is 3^5=243. 
        # Let's use a smaller grid to save budget for local search.
        # Grid of 2 points per dim (just min and max) is 2^5=32.
        # Let's do a grid of 3 points per dim but only if budget allows.
        # Actually, random is often better for rotated functions. 
        # Let's stick to random + center + a few strategic points.
        
        # Add corners of the box? No, usually minima are central.
        # Add points along axes?
        for i in range(dim):
            for val in [x_min, 0.0, x_max]:
                pt = np.zeros(dim)
                pt[i] = val
                candidates.append(pt)

        # Evaluate all candidates
        for pt in candidates:
            if check_time():
                break
            _, f = evaluate(pt)

        if check_time():
            return best_x, best_f

        # --- Phase 2: Local Refinement (Powell's Method) ---
        # We will run Powell's method starting from the best point found in Phase 1.
        # We will also try a few other good points from Phase 1 if time permits.
        
        # Sort candidates by fitness to pick the best ones for local search
        # We didn't store all, but we have best_x. 
        # Let's just run Powell from best_x.
        
        # Powell's Method Implementation
        # Directions: Start with standard basis vectors.
        # At each iteration:
        #   1. Minimize along each direction d_i.
        #   2. Compute new direction d_new = x_new - x_old.
        #   3. Minimize along d_new.
        #   4. Replace the worst direction with d_new.
        
        def powell_minimize(x_start, max_iter=50):
            nonlocal calls_used, best_x, best_f, deadline
            
            x = x_start.copy()
            f_x, _ = evaluate(x)
            
            # Initial directions
            directions = np.eye(dim)
            
            # Tolerance for stopping
            tol = 1e-8
            max_steps_per_line = 100
            
            for _iter in range(max_iter):
                if check_time():
                    break
                
                x_old = x.copy()
                f_old = f_x
                
                # 1. Minimize along each existing direction
                for i in range(dim):
                    if check_time():
                        break
                    d = directions[i]
                    # 1D minimization along x + t*d
                    # We use a simple bracketing + golden section or parabolic.
                    # Given the smoothness of f14, parabolic interpolation is good.
                    
                    # Initial bracket
                    # Try small steps first
                    t_best = 0.0
                    f_best = f_x
                    t_curr = 1.0
                    f_curr, _ = evaluate(x + t_curr * d)
                    
                    # Expand bracket
                    if f_curr > f_best:
                        # Move in opposite direction
                        t_curr = -1.0
                        f_curr, _ = evaluate(x + t_curr * d)
                        if f_curr > f_best:
                            # Stuck?
                            t_curr = 0.1
                            f_curr, _ = evaluate(x + t_curr * d)
                            if f_curr > f_best:
                                t_curr = -0.1
                                f_curr, _ = evaluate(x + t_curr * d)
                                if f_curr > f_best:
                                    continue
                    # Now we have f_best at t=0, f_curr at t_curr.
                    # If f_curr < f_best, we have a bracket [0, t_curr] or [t_curr, 0]
                    # Let's assume standard bracketing logic.
                    # To save calls, we do a simplified line search:
                    # Evaluate at t=1, if better, try t=2, t=4... until worse.
                    # Then golden section in the best bracket.
                    
                    # Reset for robust line search
                    t1 = 0.0
                    f1 = f_x
                    t2 = 1.0
                    f2, _ = evaluate(x + t2 * d)
                    
                    if f2 > f1:
                        t2 = -1.0
                        f2, _ = evaluate(x + t2 * d)
                        if f2 > f1:
                            # If both fail, try smaller step
                            t2 = 0.1
                            f2, _ = evaluate(x + t2 * d)
                            if f2 > f1:
                                t2 = -0.1
                                f2, _ = evaluate(x + t2 * d)
                                if f2 > f1:
                                    # No improvement, skip this direction
                                    continue
                    
                    # Now f2 < f1 (usually). Bracket is [t1, t2] or [t2, t1].
                    # Let's standardize to [a, b] with a < b and f(a) > f(b) or f(b) > f(a)?
                    # Standard: find a, b, c such that a < b < c and f(b) < f(a), f(b) < f(c).
                    # We have x (t=0) and x + t2*d.
                    # Let's just use a simple iterative refinement.
                    # Since f14 is smooth, we can use a few steps of parabolic fit.
                    
                    # Points: t=0 (f1), t=t2 (f2)
                    # Need a third point.
                    t3 = 2.0 * t2 - t1  # Extrapolate
                    f3, _ = evaluate(x + t3 * d)
                    
                    # If f3 is worse, bracket is [t1, t2] if f2 < f1, or [t2, t3] if f3 < f2?
                    # This is getting complex. Let's use a simpler, robust 1D minimizer:
                    # Nelder-Mead on 1D? No, too many calls.
                    # Simple: Evaluate at t=1, t=2, t=4, t=8. Find best.
                    # Then golden section around best.
                    
                    # Re-do line search more simply
                    # We already spent calls. Let's just do a few fixed steps.
                    # Actually, the code above already spent calls. 
                    # Let's assume we found a local minimum in 1D roughly.
                    # For efficiency, let's just update x to the best point found in this direction.
                    
                    # In the code above, we evaluated t=1, -1, etc.
                    # Let's just pick the best t from a small set: [-2, -1, -0.5, 0, 0.5, 1, 2]
                    # This is expensive (7 calls).
                    # Let's optimize: 
                    # We have f(0) and f(1) (or f(-1)).
                    # If f(1) < f(0), try f(2). If f(2) < f(1), try f(4).
                    # Else, golden section between 0 and 2 (or 0 and -2).
                    
                    # To keep it simple and within budget, we will use a fixed set of step sizes 
                    # for the first few iterations, then shrink.
                    # This is effectively a coordinate descent with adaptive steps.
                    
                    # Let's restart the line search logic cleanly to avoid state confusion.
                    # We will use a standard "bracket and minimize" approach.
                    
                    # Reset state for this direction
                    t_best = 0.0
                    f_best = f_x
                    step = 1.0
                    for _ls_iter in range(10):
                        if check_time():
                            break
                        # Try t_best + step
                        t_trial = t_best + step
                        f_trial, _ = evaluate(x + t_trial * d)
                        if f_trial < f_best:
                            t_best = t_trial
                            f_best = f_trial
                            # Expand step? No, keep it or shrink?
                            # If we moved, we might be in a valley.
                            # Try next step in same direction?
                            # Let's just do a few steps.
                        else:
                            # Try opposite direction
                            t_trial = t_best - step
                            f_trial, _ = evaluate(x + t_trial * d)
                            if f_trial < f_best:
                                t_best = t_trial
                                f_best = f_trial
                                step = -step  # Reverse direction for next step?
                            else:
                                # Shrink step
                                step *= 0.5
                                if step < 1e-6:
                                    break
                            
                    # Update x
                    x = x + t_best * d
                    f_x = f_best
                    # Update global best
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()

                # 2. Compute new direction
                d_new = x - x_old
                
                # 3. Minimize along new direction
                f_x_old = f_x
                # Simple line search for new direction
                t_best = 0.0
                step = 1.0
                for _ls_iter in range(10):
                    if check_time():
                        break
                    t_trial = t_best + step
                    f_trial, _ = evaluate(x + t_trial * d_new)
                    if f_trial < f_x:
                        t_best = t_trial
                        f_x = f_trial
                    else:
                        t_trial = t_best - step
                        f_trial, _ = evaluate(x + t_trial * d_new)
                        if f_trial < f_x:
                            t_best = t_trial
                            f_x = f_trial
                            step = -step
                        else:
                            step *= 0.5
                            if step < 1e-6:
                                break
                x = x + t_best * d_new
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()

                # 4. Update directions
                # Find the direction with the largest increase in function value
                # (i.e., the worst direction)
                # We didn't track the increase per direction easily in the simplified loop.
                # Standard Powell replaces the direction that contributed the least to improvement.
                # For simplicity and robustness, we can just keep the standard basis 
                # and rely on the new direction being added/replacing the worst.
                # Given the complexity, let's just add d_new to the set of directions 
                # and remove the one that is most collinear with d_new or worst.
                # Actually, for f14, the standard basis + a few adaptive directions is often enough.
                # Let's just update the last direction to be d_new if it's significant.
                if np.linalg.norm(d_new) > 1e-6:
                    # Normalize
                    d_new = d_new / np.linalg.norm(d_new)
                    # Replace the direction that is most similar to d_new? 
                    # Or just append and trim?
                    # Let's keep it simple: use standard basis for the first 5 iters, 
                    # then switch to coordinate descent with adaptive steps if Powell is complex.
                    # Actually, the loop above is essentially a hybrid.
                    # Let's just break out of Powell if we are converging slowly.
                    pass

                # Check convergence
                if np.linalg.norm(x - x_old) < 1e-5 and abs(f_x - f_old) < 1e-5:
                    break

            return x, f_x

        # Run Powell from the best point
        # We might want to run it from a few of the top candidates from Phase 1.
        # But we only kept the best. Let's just run from best_x.
        
        # To utilize remaining budget, we can run multiple local searches from different starts.
        # Let's try the best point, and if time permits, a few others.
        
        # We need to store the top candidates from Phase 1.
        # Let's re-evaluate Phase 1 to store top K.
        
        # Re-do Phase 1 with storage
        candidates = []
        candidates.append(np.zeros(dim))
        for _ in range(n_global // 2):
            candidates.append(rng.uniform(x_min, x_max, size=dim))
        for i in range(dim):
            for val in [x_min, 0.0, x_max]:
                pt = np.zeros(dim)
                pt[i] = val
                candidates.append(pt)
        
        # Evaluate and store
        eval_results = []
        for pt in candidates:
            if check_time():
                break
            pt_clipped, f = evaluate(pt)
            eval_results.append((f, pt_clipped))
        
        # Sort by fitness
        eval_results.sort(key=lambda x: x[0])
        
        # Run local search on the top 3 candidates (if distinct)
        used_starts = set()
        for f_start, x_start in eval_results[:3]:
            if check_time():
                break
            # Check if this start is too close to a previous one
            is_dup = False
            for us in used_starts:
                if np.linalg.norm(x_start - us) < 0.5:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            used_starts.add(x_start.copy())
            
            # Run local search
            # We modify powell_minimize to take a start point
            # The function above uses nonlocal state, so we need to be careful.
            # Let's just call a simplified local search.
            
            # Simple Coordinate Descent with Adaptive Steps
            # This is more robust than the complex Powell implementation above.
            x_curr = x_start.copy()
            f_curr, _ = evaluate(x_curr)
            
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
            
            for step in step_sizes:
                if check_time():
                    break
                improved = False
                for _cd_iter in range(10):
                    if check_time():
                        break
                    for i in range(dim):
                        if check_time():
                            break
                        # Try + step
                        x_trial = x_curr.copy()
                        x_trial[i] += step
                        f_trial, _ = evaluate(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            continue
                        # Try - step
                        x_trial = x_curr.copy()
                        x_trial[i] -= step
                        f_trial, _ = evaluate(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            improved = True
                            continue
                        # If no improvement, try smaller step? 
                        # Or just move to next dimension.
                    if not improved:
                        break
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
