import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Keep 30s margin for safety
        deadline = start_time + 290 
        
        n = self.dim
        budget = self.budget
        
        # 1. Multi-start Nelder-Mead Strategy
        # Rastrigin is multimodal. A single start is risky.
        # We divide the budget among several independent Nelder-Mead runs.
        # NM is very fast per evaluation and good at local refinement.
        
        # Heuristic: 5-10 restarts is often sufficient for dim=5 Rastrigin.
        # Let's use 8 restarts.
        num_restarts = 8
        budget_per_restart = budget // num_restarts
        
        # To ensure early progress for AOCC, we start with a few random evaluations
        # to get a baseline, then launch the restarts.
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Generate initial starting points
        # Mix of center, random, and grid-like points to cover the space
        start_points = []
        
        # 1. Center
        start_points.append(np.zeros(n))
        
        # 2. Random points
        for _ in range(num_restarts - 1):
            p = rng.uniform(-5, 5, n)
            start_points.append(p)
            
        # Shuffle to randomize order? No, keep deterministic order for reproducibility
        # but the randomness comes from the initial seed.
        
        # We will iterate through these start points.
        # If we run out of budget, we stop.
        
        # To maximize AOCC, we want to find a good solution quickly.
        # Nelder-Mead converges fast.
        
        # We will manage the budget carefully.
        # Each NM call needs a wrapper to count evaluations.
        
        for i, x0 in enumerate(start_points):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            # Allocate remaining budget to this restart, but cap it to allow other restarts if possible
            # Actually, if we have 8 restarts, and we give each 1/8th, we might underutilize if one converges fast.
            # Better strategy: Dynamic allocation.
            # Give each restart a "chunk" of the budget.
            
            remaining_budget = budget - calls_made
            # If this is the last restart, give it all.
            # Otherwise, give it a fixed chunk.
            if i == len(start_points) - 1:
                max_iter = remaining_budget
            else:
                # Leave some budget for subsequent restarts
                # A simple heuristic: 1/num_restarts of total budget, or remaining if less
                # But NM can stop early if it converges.
                max_iter = min(remaining_budget, budget_per_restart)
                
            if max_iter < 10:
                continue

            # Wrapper for NM
            nm_calls = [0]
            def nm_func(x_nm):
                nonlocal calls_made, best_f, best_x
                if nm_calls[0] >= max_iter or calls_made >= budget or time.time() > deadline:
                    # Return a value that discourages further exploration if we hit limit
                    # But NM might still call. We just want to stop.
                    # Returning best_f is safe but might bias. 
                    # Actually, if we hit budget, we should stop.
                    # The `maxiter` option in scipy handles the internal loop limit.
                    # But we need to ensure we don't exceed the global budget.
                    return best_f
                
                nm_calls[0] += 1
                calls_made += 1
                
                # Clip to box
                x_clipped = np.clip(x_nm, -5, 5)
                f_val = func(x_clipped)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                    
                return f_val

            try:
                # Nelder-Mead parameters
                # xatol and fatol control convergence
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                
                # Update best if res.fun is better (it should be, via wrapper, but double check)
                if res.fun < best_f:
                    best_f = res.fun
                    # res.x might be outside box? No, we clipped in wrapper.
                    # But res.x is the last point evaluated.
                    # We should ensure best_x corresponds to best_f.
                    # The wrapper updates best_x when best_f is updated.
                    pass
                    
            except Exception:
                pass

        # If we still have budget left, do a final polish on the best point found
        if calls_made < budget and time.time() < deadline and best_x is not None:
            remaining = budget - calls_made
            if remaining > 20:
                nm_calls = [0]
                def polish_func(x_nm):
                    nonlocal calls_made, best_f, best_x
                    if nm_calls[0] >= remaining or calls_made >= budget or time.time() > deadline:
                        return best_f
                    nm_calls[0] += 1
                    calls_made += 1
                    x_clipped = np.clip(x_nm, -5, 5)
                    f_val = func(x_clipped)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x_clipped.copy()
                    return f_val
                
                try:
                    res = minimize(polish_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                except Exception:
                    pass

        return best_x, best_f
