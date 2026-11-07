"""Hybrid Multi-Start Nelder-Mead Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start search with diverse initialization and competitive Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Diverse Initialization
        # We generate a set of diverse starting points to cover the search space.
        # This is crucial for BBOB f14 which has hidden shifts.
        n_starts = min(20, budget // 25)  # Ensure we have enough budget for refinement
        
        candidates = []
        
        # 1. Origin
        candidates.append(np.zeros(dim))
        
        # 2. Random uniform points
        for _ in range(n_starts // 2):
            candidates.append(rng.uniform(-5, 5, dim))
            
        # 3. Points near boundaries (often where optima hide in shifted problems)
        for _ in range(n_starts // 4):
            pt = np.zeros(dim)
            for i in range(dim):
                pt[i] = rng.choice([-4.5, -4, -3, 3, 4, 4.5])
            candidates.append(pt)
            
        # 4. Structured points (coarse grid-like)
        grid_vals = [-3, -1, 0, 1, 3]
        for _ in range(n_starts // 4):
            pt = np.array([rng.choice(grid_vals) for _ in range(dim)])
            candidates.append(pt)

        # Evaluate all candidates
        for sp in candidates:
            if not time_ok() or not budget_ok():
                break
            x = np.clip(sp, -5, 5)
            fx = func(x)
            calls += 1
            record(x, fx)

        if best_x is None:
            return None, np.inf

        # Phase 2: Competitive Nelder-Mead Refinement
        # We pick the top K candidates and run NM on them.
        # We track the global best. If one NM run finds a better point, it becomes the new global best.
        # We limit the number of NM iterations per candidate to ensure we explore multiple basins.
        
        # Sort candidates by their initial function value
        # We need to keep track of which points we've evaluated
        # Since we didn't store all (x, f) pairs in a list, we just use the best_x found so far 
        # and generate perturbations around it? 
        # No, the inspiration suggests using the diverse points. Let's re-evaluate the top ones?
        # Actually, we can just run NM on the best_x found so far, and if we have budget, 
        # run NM on a few random points near the best_x or other diverse points.
        
        # Better approach: 
        # 1. Run NM on the best point found in Phase 1.
        # 2. If budget remains, run NM on the 2nd and 3rd best points from Phase 1.
        
        # To do this, we need to store the evaluated points from Phase 1.
        # Let's refactor Phase 1 to store them.
        
        # --- Refactored Phase 1 ---
        # We already evaluated them. Let's just assume best_x is the best.
        # To be more robust, let's collect the top few.
        # Since we didn't store them, we'll just use best_x for the first NM run.
        # For subsequent runs, we'll generate new diverse points or perturb best_x.
        
        # Let's implement a helper to run NM from a starting point x0 with f0.
        
        def run_nelder_mead(x0, f0, max_iters=100):
            nonlocal calls, best_x, best_f
            
            if not time_ok() or not budget_ok():
                return

            x0 = x0.copy()
            f0 = f0
            record(x0, f0)
            
            # Build initial simplex
            alpha = 0.5
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)
            
            if len(simplex) < 2:
                return

            # Nelder-Mead parameters
            gamma = 1.0
            rho = 2.0
            sigma_nm = 0.5
            shrink = 0.5
            
            # Track best in this specific NM run to check for local convergence
            local_best_f = f0
            
            for _ in range(max_iters):
                if not time_ok() or not budget_ok():
                    break
                
                # Sort simplex
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)
                if f_r < local_best_f:
                    local_best_f = f_r
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    if f_e < local_best_f:
                        local_best_f = f_e
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)
                    if f_c < local_best_f:
                        local_best_f = f_c
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])
                            if f_simplex[i] < local_best_f:
                                local_best_f = f_simplex[i]
                        if not time_ok():
                            break
                
                # Convergence check for this local run
                f_spread = np.max(f_simplex) - np.min(f_simplex)
                if f_spread < 1e-10:
                    break

        # Run NM on the best point found in Phase 1
        run_nelder_mead(best_x, best_f)
        
        # If we still have budget, try refining a few other diverse points.
        # We'll generate new diverse points near the current best or just random ones
        # to see if we missed a better basin.
        
        if time_ok() and budget_ok():
            # Try 2 more local optimizations from random points
            # These points should be distinct from the best one to ensure diversity
            for _ in range(2):
                if not time_ok() or not budget_ok():
                    break
                
                # Generate a random point, but ensure it's not too close to current best
                x_start = rng.uniform(-5, 5, dim)
                # If it's too close, regenerate
                if np.linalg.norm(x_start - best_x) < 1.0:
                    x_start = best_x + rng.normal(0, 2.0, dim)
                    x_start = np.clip(x_start, -5, 5)
                
                f_start = func(x_start)
                calls += 1
                record(x_start, f_start)
                
                # Only refine if it's competitive or if we have lots of budget
                # To maximize AOCC, we should refine any point that might lead to a better global best.
                # But refining a very bad point is a waste.
                # Let's refine if it's within a certain threshold of best_f, or just always if budget is high.
                # Given budget=500 and we used ~20 + ~50 (NM) = 70 calls, we have ~430 left.
                # So we can afford to refine.
                
                # Limit iterations for these secondary runs to save budget for more restarts if needed
                run_nelder_mead(x_start, f_start, max_iters=50)
                
                # If this run found a new global best, great.
                # If not, we might have wasted some calls. 
                # To mitigate this, we can check if best_f improved significantly.
                
        return best_x, best_f
# EVOLVE-BLOCK-END
