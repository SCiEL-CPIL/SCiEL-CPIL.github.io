import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Quasi-random global search with Powell's local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # 1. Global Exploration using Sobol Sequence (Quasi-Random)
        n_global = 100
        if budget < n_global:
            n_global = budget
            
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Generate points in [0, 1]^dim
        points_unit = sampler.random(n=n_global)
        # Map to [-5, 5]^dim
        points = points_unit * 10.0 - 5.0
        
        calls_made = 0
        best_f = np.inf
        best_x = None
        candidate_starts = []
        
        for i in range(n_global):
            x = points[i]
            fx = func(x)
            calls_made += 1
            candidate_starts.append((fx, x))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if calls_made >= budget:
            return best_x, best_f

        # 2. Local Refinement using Powell's Method
        K = 5
        candidate_starts.sort(key=lambda t: t[0])
        top_starts = [x for _, x in candidate_starts[:K]]
        
        remaining_budget = budget - calls_made
        budget_per_start = remaining_budget // K
        
        for start_x in top_starts:
            if budget_per_start <= 0:
                break
            
            # Wrapper to strictly enforce budget if needed, 
            # though scipy's maxfev is usually sufficient.
            # To be absolutely safe, we can use a closure.
            local_calls = 0
            limit = budget_per_start
            
            def wrapped_func(x):
                nonlocal local_calls
                local_calls += 1
                if local_calls > limit:
                    raise ValueError("Budget exceeded")
                return func(x)
            
            try:
                res = minimize(wrapped_func, start_x, method='Powell', 
                               options={'maxfev': limit, 'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8})
                
                # Update best if improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If an exception occurs (e.g., budget exceeded or convergence issue),
                # we keep the best found so far and continue to next start if budget remains.
                # The task says: "an exception later keeps the calls made".
                # So we just break out of this local search and try the next one if possible.
                pass

        return best_x, best_f
