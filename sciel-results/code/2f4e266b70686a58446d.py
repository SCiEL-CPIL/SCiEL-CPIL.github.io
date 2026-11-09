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
        # We use 60 points for initial spread + center point
        n_lhs = 60
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
        
        # Also evaluate the center point (0,0,0,0,0) as it's near the unshifted minimum
        if calls_made < budget:
            x_center = np.zeros(dim)
            fx_center = func(x_center)
            calls_made += 1
            update_best(x_center, fx_center)

        # Phase 2: Multi-Start Local Search
        # We divide the remaining budget into trials.
        # Each trial consists of a "Jump" from the current best (or random) 
        # followed by a local (1+1)-ES refinement.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # No reserved polish budget - let ES use most of it, then final NM with whatever's left
        search_budget = remaining_budget
        
        # Define trial structure with adaptive length
        base_trial_length = 40
        max_trial_length = 60
        min_trial_length = 20

        sigma_init = 2.0
        min_sigma = 0.003  # Even finer for better convergence
        max_sigma = 5.0
        jump_scale = 3.0 # Scale of jump from best to escape basin

        trial_idx = 0
        while calls_made < budget - 15:  # Reserve at least 15 for NM polish
            if time.time() > deadline:
                return x_global_best, fx_global_best
            
            # Determine starting point for this trial
            if trial_idx == 0:
                # First trial: Start from the best LHS point
                x = x_global_best.copy()
            elif trial_idx == 1:
                # Second trial: Try a random point far from current best
                x = rng.uniform(bounds, upper)
            else:
                # Subsequent trials: Jump from global best to explore new basin
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            
            # Local (1+1)-ES Phase
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            no_progress_count = 0
            max_no_progress = 15
            
            # Run local search with adaptive stopping
            while calls_made < budget - 15 and no_progress_count < max_no_progress:
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
                    sigma = min(sigma * 1.2, max_sigma)
                    update_best(x, fx)
                    no_progress_count = 0
                else:
                    # Decrease sigma on failure to refine
                    sigma = max(sigma * 0.8, min_sigma)
                    no_progress_count += 1
                    
            trial_idx += 1
        # Phase 3: Nelder-Mead Polish
        # Use the remaining budget to polish the global best point
        remaining = budget - calls_made
        if remaining > 10 and x_global_best is not None:
            # Simple Nelder-Mead implementation
            simplex = [x_global_best.copy()]
            f_simplex = [fx_global_best]
            
            # Create initial simplex with very small perturbations for fine convergence
            perturbation_size = 0.05
            for i in range(dim):
                perturbation = x_global_best.copy()
                if perturbation[i] > bounds[i] + perturbation_size:
                    perturbation[i] -= perturbation_size
                else:
                    perturbation[i] += perturbation_size
                perturbation = np.clip(perturbation, bounds, upper)
                fx_pert = func(perturbation)
                calls_made += 1
                simplex.append(perturbation)
                f_simplex.append(fx_pert)
                update_best(perturbation, fx_pert)
            
            # Nelder-Mead iterations with tighter convergence
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5 # shrinkage
            
            while calls_made < budget - 2 and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check for convergence
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-12:
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
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break
            
            # Final update
            for i in range(len(simplex)):
                update_best(simplex[i], f_simplex[i])

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
