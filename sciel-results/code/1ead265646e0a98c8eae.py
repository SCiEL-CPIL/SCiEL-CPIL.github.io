"""Optimizer for BBOB f23 (Katsuura) using Multi-Start (1+1)-ES with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with LHS initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # We use 50 points for initial spread
        n_lhs = 50
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        # Phase 2: Multi-Start Local Search
        # We divide the remaining budget into trials.
        # Each trial consists of a "Jump" from the current best (or random) 
        # followed by a local (1+1)-ES refinement.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Reserve budget for final polishing phase
        reserve_polish = min(30, remaining_budget // 5)
        search_budget = remaining_budget - reserve_polish
        
        # Define trial structure
        # We want roughly 40-50 steps per trial to allow convergence.
        trial_length = 40
        n_trials = search_budget // trial_length
        
        # If we can't do a full trial, do one big remaining chunk
        if n_trials == 0:
            trial_length = search_budget
            n_trials = 1

        sigma_init = 2.0
        min_sigma = 0.005  # Lower min_sigma for finer convergence
        max_sigma = 5.0
        jump_scale = 3.0 # Scale of jump from best to escape basin

        for t in range(n_trials):
            if calls_made >= budget - reserve_polish:
                break
            
            # Determine starting point for this trial
            # Strategy: Jump from global best with some randomness
            # If this is the first trial after LHS, we can also try a pure random point 
            # to ensure we don't just refine the LHS best too much if it's bad.
            if t == 0:
                # First trial: Start from the best LHS point, but maybe jump a bit?
                # Or just refine it. Let's refine it first.
                x = x_global_best.copy()
            else:
                # Subsequent trials: Jump from global best to explore new basin
                # The jump should be large enough to escape the current local min
                # but not so large it goes out of bounds (handled by clip)
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            
            # Local (1+1)-ES Phase
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            
            # Run local search for up to trial_length steps
            for step in range(trial_length - 1): # -1 because we already evaluated x
                if calls_made >= budget - reserve_polish:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    # Increase sigma on success to move faster
                    sigma = min(sigma * 1.25, max_sigma)
                    update_best(x, fx)
                else:
                    # Decrease sigma on failure to refine
                    sigma = max(sigma * 0.75, min_sigma)
                    
        # Phase 3: Nelder-Mead Polish
        # Use the remaining budget to polish the global best point
        # This exploits the smooth structure for fine-grained progress
        remaining = budget - calls_made
        if remaining > 10 and x_global_best is not None:
            # Simple Nelder-Mead implementation
            simplex = [x_global_best.copy()]
            f_simplex = [fx_global_best]
            
            # Create initial simplex with smaller perturbations for better convergence
            for i in range(dim):
                perturbation = x_global_best.copy()
                if perturbation[i] > bounds[i] + 0.1:
                    perturbation[i] -= 0.2
                else:
                    perturbation[i] += 0.2
                perturbation = np.clip(perturbation, bounds, upper)
                fx_pert = func(perturbation)
                calls_made += 1
                simplex.append(perturbation)
                f_simplex.append(fx_pert)
                update_best(perturbation, fx_pert)
            
            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5 # shrinkage
            
            while calls_made < budget - 5 and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check for convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                
                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, bounds, upper)
                fx_r = func(x_r)
                calls_made += 1
                update_best(x_r, fx_r)
                
                if fx_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, bounds, upper)
                    fx_e = func(x_e)
                    calls_made += 1
                    update_best(x_e, fx_e)
                    
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = fx_r
                elif fx_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, bounds, upper)
                    fx_c = func(x_c)
                    calls_made += 1
                    update_best(x_c, fx_c)
                    
                    if fx_c < min(fx_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = fx_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds, upper)
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            update_best(simplex[i], f_simplex[i])
            
            # Update global best with the best point in the final simplex
            for i in range(len(simplex)):
                update_best(simplex[i], f_simplex[i])

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
