import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin

        # Helper: Check if we are out of time or budget
        def check_stop(evals):
            return evals >= budget or time.time() > deadline

        # 1. Global Exploration using Sobol Sequence (or Halton if Sobol not available)
        # We use the first 50 points for global search.
        n_global = min(50, budget)
        
        # Generate Sobol points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            points_unit = sampler.random(n_global)
        except ImportError:
            # Fallback to Halton
            def halton(dim, n):
                points = np.zeros((n, dim))
                for d in range(dim):
                    base = d + 2
                    f = 1.0
                    i = 1
                    for k in range(n):
                        f /= base
                        i += 1
                        while i % base == 0:
                            i //= base
                            f *= base
                        points[k, d] = f
                return points
            points_unit = halton(dim, n_global)

        # Map [0, 1]^dim to [-5, 5]^dim
        X_global = points_unit * 10.0 - 5.0
        
        F_global = np.zeros(n_global)
        evals = 0
        
        for i in range(n_global):
            if check_stop(evals):
                break
            x = X_global[i]
            F_global[i] = func(x)
            evals += 1
        
        # Find the best point from global search
        best_idx = np.argmin(F_global[:evals])
        best_x = X_global[best_idx].copy()
        best_f = F_global[best_idx]
        
        # Sort global points by fitness to use as potential restarts
        # We only care about the top few
        valid_indices = np.argsort(F_global[:evals])
        
        # 2. Local Optimization using Nelder-Mead Simplex
        # We will attempt to refine the best point, and if it stalls, try the next best.
        
        def nelder_mead(func, x0, evals, deadline, budget, dim, max_iter=1000):
            """
            Implements Nelder-Mead Simplex method.
            Returns (best_x, best_f, evals_used)
            """
            # Initial simplex
            # Start with a small simplex around x0
            # Step size: 5% of the box width (10.0) -> 0.5
            step = 0.5 
            
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0
                simplex[i, i-1] += step
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if check_stop(evals):
                    break
                simplex_f[i] = func(simplex[i])
                evals += 1
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            best_x = simplex[0].copy()
            best_f = simplex_f[0]
            
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrink
            
            for _ in range(max_iter):
                if check_stop(evals):
                    break
                
                # Check for convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                    break
                
                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                
                f_ref = func(x_ref)
                evals += 1
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = func(x_exp)
                    evals += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Reflection is good but not better than best
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    f_con = func(x_con)
                    evals += 1
                    
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if check_stop(evals):
                                break
                            simplex_f[i] = func(simplex[i])
                            evals += 1
                
                # Re-sort simplex
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
            
            return best_x, best_f, evals

        # Strategy: Try the top 3 global points with Nelder-Mead
        # Allocate budget dynamically.
        # If we have ~450 calls left, we can run NM on the best.
        # If it stalls, we might have calls left for the next.
        
        candidates = [X_global[i] for i in valid_indices[:min(3, evals)]]
        
        for cand in candidates:
            if check_stop(evals):
                break
            
            # Skip if this candidate is not better than current best (shouldn't happen as we sorted)
            # But if we already improved best_f significantly, and this candidate is far, maybe skip?
            # For now, just run NM.
            
            # We want to ensure we don't spend ALL budget on the first candidate if it fails to improve much.
            # However, NM usually converges quickly.
            
            x_res, f_res, evals = nelder_mead(func, cand, evals, deadline, budget, dim)
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()
                
            # If we have used most of the budget, stop
            if evals > budget * 0.8:
                break

        # 3. Final Polish: Coordinate Descent with small steps
        # If we have time and budget left, do a fine-tune
        if evals < budget and time.time() < deadline:
            step = 0.05
            x = best_x.copy()
            f = best_f
            no_improve = 0
            max_no_improve = 5
            
            while evals < budget and time.time() < deadline and no_improve < max_no_improve:
                improved = False
                for i in range(dim):
                    if check_stop(evals):
                        break
                    for direction in [-1, 1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                    if check_stop(evals):
                        break
                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-6:
                        break
                else:
                    no_improve = 0
                    # Keep step or slightly increase if we are making good progress?
                    # Usually decreasing is safer for final polish.
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
