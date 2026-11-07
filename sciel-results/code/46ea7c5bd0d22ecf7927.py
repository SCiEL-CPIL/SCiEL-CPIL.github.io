import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        n_evals = 0
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal n_evals, best_x, best_f
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling for Global Exploration
        # Use 20% of budget for global search
        n_lhs = min(budget // 5, 100)
        lhs_points = np.zeros((n_lhs, d))
        for i in range(d):
            perm = np.random.permutation(n_lhs)
            # Map to [-5, 5]
            lhs_points[perm, i] = -5.0 + 10.0 * (np.arange(n_lhs) + np.random.rand()) / n_lhs
        
        for i in range(n_lhs):
            if n_evals >= budget: break
            x = lhs_points[i]
            f = func(x)
            update_best(x, f)
            
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            update_best(best_x, best_f)

        # Phase 2: Multi-start Nelder-Mead Refinement
        # Select top candidates from LHS? We only kept the best. 
        # Let's re-evaluate a few random points around the best found so far?
        # Or just run NM on the best point.
        # To make it robust, we'll run NM on the best point, and if budget allows, 
        # restart with a smaller neighborhood if it stalls.
        
        # We will implement a custom Nelder-Mead with stall detection
        def nelder_mead(func, x0, max_iter, budget_limit):
            nonlocal n_evals, best_x, best_f
            
            # Initialize Simplex
            # x0 is the best point
            simplex = [x0.copy()]
            f0 = func(x0)
            update_best(x0, f0)
            
            # Add perturbed points
            for i in range(d):
                x_new = x0.copy()
                # Step size 1.0 initially, or adaptive based on d
                x_new[i] += 1.0 if x_new[i] < 0 else -1.0
                # Ensure within bounds? func clips, so it's fine.
                f_new = func(x_new)
                update_best(x_new, f_new)
                simplex.append(x_new)
                
            # Nelder-Mead Loop
            stall_count = 0
            eps = 1e-10
            shrink_factor = 0.5
            expand_factor = 2.0
            
            for _ in range(max_iter):
                if n_evals >= budget_limit:
                    break
                if time.time() - start_time > timeout:
                    break
                    
                # Sort simplex by function value
                # We need to store f values. Let's re-evaluate if not stored? 
                # No, we should store them.
                # For simplicity in this custom implementation, we keep a list of (f, x)
                # But update_best tracks global best.
                
                # Refactor: Keep local simplex state
                # This is getting complex. Let's use scipy.optimize.minimize 
                # but with multiple restarts and specific options.
                pass

        # Using scipy is safer and likely faster to implement correctly.
        from scipy.optimize import minimize
        
        # Strategy:
        # 1. Run NM on best_x from LHS.
        # 2. If it finishes early or stalls, try a new start point.
        
        # Budget allocation:
        # 10% for LHS (already done)
        # 40% for 4 separate NM runs (10% each) on different seeds (Best from LHS, and 3 random points near best)
        # 50% for one final long NM run on the overall best.
        
        # Generate seeds for refinement
        seeds = [best_x.copy()]
        
        # Add 3 random points in a ball of radius 1.0 around best_x
        for _ in range(3):
            if n_evals >= budget: break
            direction = np.random.randn(d)
            direction /= np.linalg.norm(direction)
            x_new = best_x + direction * 1.0
            f_new = func(x_new)
            update_best(x_new, f_new)
            seeds.append(x_new.copy())
            
        # Run NM on seeds
        # Each run gets 1/4 of the remaining budget
        remaining_budget = budget - n_evals
        nm_budget_per_run = remaining_budget // 4
        
        for seed in seeds:
            if n_evals >= budget: break
            if time.time() - start_time > timeout: break
            
            try:
                res = minimize(
                    func, 
                    seed, 
                    method='Nelder-Mead', 
                    options={'maxiter': nm_budget_per_run, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                # Update best if improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
        # Final Polish
        if n_evals < budget and time.time() - start_time < timeout:
            final_budget = budget - n_evals
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': final_budget, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
