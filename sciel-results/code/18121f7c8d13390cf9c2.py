import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization
        # Sample 10 points to identify the best basin.
        # Step-ellipsoidal functions often have a "step" where the landscape changes.
        # Finding the right side of the step is crucial.
        init_candidates = []
        for _ in range(10):
            if evals_used >= budget: break
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x))
        
        init_candidates.sort(key=lambda k: k[0])
        x_start = init_candidates[0][1].copy()
        best_f = init_candidates[0][0]

        # Phase 2: Multi-Start Nelder-Mead
        # We use the top 5 initial candidates as seeds for 5 independent NM runs.
        # This exploits the step-ellipsoidal structure by rapidly contracting simplices.
        # The remaining budget is divided among these runs.
        
        # We will run NM on the top 5 distinct points (or fewer if not enough)
        # Actually, let's just run NM on the best point found so far, but with a robust restart mechanism?
        # The directive suggests "Multi-start Nelder-Mead simplex method with 10 diverse initial centroids".
        # Let's implement a loop that takes the best point, runs NM for a chunk of budget, 
        # and if it stagnates, restarts from a perturbed version or a different initial point.
        
        # However, a simpler and often more effective strategy for this specific task (f7) 
        # is to run CMA-ES for a short period to get a good basin, then switch to NM.
        # But the directive specifically asks to combine with the inspiration which was NM-heavy.
        
        # Let's do this:
        # 1. Use the best of the 10 init points.
        # 2. Run Nelder-Mead for the remaining budget, but with a "restart" strategy if it gets stuck.
        # Actually, running 5 separate NM instances in parallel (interleaved) is risky for budget management.
        # Let's run a single, robust NM that can restart.
        
        # Better approach based on inspiration:
        # Use the first 10 evals to seed.
        # Then, run NM starting from the best seed.
        # If NM stagnates (no improvement for X evals), restart from a random perturbation of the best found so far.
        
        def nelder_mead(x0, f0, max_evals):
            nonlocal best_x, best_f, evals_used
            
            # Initialize simplex
            # Simplex has dim+1 points
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            # Expand simplex in each dimension
            step_size = 0.5 # Initial step for simplex expansion
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += step_size
                f_new = eval_point(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
            
            # Track best in simplex
            best_idx = 0
            for i in range(1, dim + 1):
                if f_simplex[i] < f_simplex[best_idx]:
                    best_idx = i
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            no_improve_count = 0
            last_best_f = best_f
            
            while evals_used < budget and (budget - evals_used) > 0:
                if time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Update global best
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check stagnation
                if no_improve_count > 20:
                    # Restart: perturb the best point
                    # Break out of current NM loop and let outer logic handle restart?
                    # Or just restart internally?
                    # Let's break and let the main loop restart with a new seed.
                    break
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_point(x_r)
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = eval_point(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    f_c = eval_point(x_c)
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = eval_point(simplex[i])
                        
                        # If we used too many evals in shrink, we might have hit budget
                        if evals_used >= budget:
                            break
            
            return best_x, best_f

        # Main Loop: Multi-Start / Restart Strategy
        # We will attempt up to 5 restarts.
        # Start with the best init point.
        # If NM stagnates, pick the 2nd best init point, or a random perturbation.
        
        # List of seeds: Top 5 from init, plus some random perturbations of the best
        seeds = [x_start.copy()]
        for i in range(1, 5):
            seeds.append(init_candidates[i][1].copy())
        
        # Add some random perturbations of the best found so far
        for _ in range(5):
            x_pert = best_x + rng.normal(0, 1.0, dim)
            seeds.append(x_pert)
        
        # Shuffle seeds to ensure diversity in restart order? 
        # No, we want to try the best ones first.
        
        # We will run NM on each seed if we have budget left.
        # To avoid wasting budget on bad seeds, we limit each NM run to a portion of the remaining budget.
        # But NM is adaptive. Let's just let it run until it stagnates or budget is low.
        
        for seed in seeds:
            if evals_used >= budget - 10: # Need some evals left to finish
                break
            if time.time() > deadline:
                break
            
            # Ensure seed is not the same as current best (to avoid immediate stagnation)
            # If it is the same, we might just continue, but let's force a slight perturbation if it's the first seed
            if np.allclose(seed, best_x):
                # Perturb slightly
                seed = seed + rng.normal(0, 0.1, dim)
            
            f_seed = eval_point(seed)
            
            # Run NM
            x_res, f_res = nelder_mead(seed, f_seed, budget - evals_used)
            
            # If we improved, update seeds? 
            # The loop structure handles the next seed.
            
        # Phase 3: Final Local Refinement
        # Use remaining budget for a (1+1)-ES or coordinate descent
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Adaptive step size
            h = 0.1
            
            for _ in range(remaining):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(dim):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_try = x_local.copy()
                    x_try[i] += h
                    f_try = eval_point(x_try)
                    
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        improved = True
                        h *= 1.1 
                    else:
                        # Try negative step
                        x_try = x_local.copy()
                        x_try[i] -= h
                        f_try = eval_point(x_try)
                        
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                            h *= 1.1 
                        else:
                            # Shrink step
                            h *= 0.7
                            
                    if h < 1e-5:
                        h = 0.05 # Reset if too small
                            
                if not improved and h < 0.01:
                    # Try a small random perturbation to escape local minima
                    x_try = x_local + rng.normal(0, 0.1, dim)
                    f_try = eval_point(x_try)
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        h = 0.1
                    else:
                        break # Stagnant
            
            # Update best
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
