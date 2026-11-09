"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Global L-BFGS-B optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        dim = self.dim
        budget = self.budget
        
        def check_time():
            return time.time() > deadline

        def wrapper(x):
            nonlocal calls_used
            if check_time():
                return np.inf
            fx = func(x)
            calls_used += 1
            return fx

        # Strategy: Multi-start L-BFGS-B
        # Rastrigin is multimodal. We need to sample the basin of attraction of the global minimum.
        # L-BFGS-B is very fast in low dimensions (5D) and uses gradient information implicitly 
        # (via finite differences in scipy, or we could use analytic if we knew the function, 
        # but we only have a black box). 
        # Since we don't have analytic gradients, scipy's L-BFGS-B uses numerical gradients.
        # This is expensive (N+1 calls per step), but it converges very quickly to local minima.
        # 
        # Budget allocation:
        # Total budget 500.
        # Each L-BFGS-B run might take 20-50 function evaluations to converge to a local min.
        # We can afford roughly 10-20 independent restarts.
        
        # Generate diverse starting points
        # 1. Origin (good for Rastrigin if shift is small)
        # 2. Random points uniformly in [-5, 5]
        # 3. Random points in a smaller box around 0, e.g. [-2, 2], to catch the main basin if shift is small
        
        start_points = []
        start_points.append(np.zeros(dim))
        
        # Add random points
        # We want to cover the space. 
        # Let's use 15 random starts.
        for _ in range(15):
            # Mix of wide and narrow distributions
            if rng.random() < 0.5:
                p = rng.uniform(-5, 5, dim)
            else:
                p = rng.uniform(-2, 2, dim)
            start_points.append(p)

        # Sort start points to try promising ones first? 
        # Origin is usually good. Then random.
        
        for x0 in start_points:
            if calls_used >= budget or check_time():
                break
            
            # Define bounds
            bounds = [(-5.0, 5.0) for _ in range(dim)]
            
            # Run L-BFGS-B
            # maxiter: limit iterations to prevent runaway if it doesn't converge
            # maxfun: limit function calls per run
            # We have remaining budget. Let's give each run a fair share or let it run until convergence.
            # If it converges quickly, we save budget for more restarts.
            remaining_budget = budget - calls_used
            
            # Cap function calls per run to ensure we get multiple restarts
            # 500 calls / 16 starts ~ 30 calls/start. 
            # But early starts might be "cheap" if they converge fast.
            # Let's allow up to 50 calls per run, but stop if budget runs out.
            max_fun_per_run = min(50, remaining_budget)
            
            if max_fun_per_run <= 0:
                break

            try:
                res = minimize(wrapper, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'maxfun': max_fun_per_run})
                
                # Update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If L-BFGS-B fails, we just move to the next start
                pass

        # If we have budget left and time, do a final polish on the best point found
        if best_x is not None and calls_used < budget and not check_time():
             remaining_budget = budget - calls_used
             if remaining_budget > 10:
                 try:
                     res = minimize(wrapper, best_x, method='L-BFGS-B', bounds=[(-5.0, 5.0) for _ in range(dim)], 
                                    options={'maxiter': 50, 'maxfun': remaining_budget})
                     if res.fun < best_f:
                         best_f = res.fun
                         best_x = res.x.copy()
                 except Exception:
                     pass

        return best_x, best_f
# EVOLVE-BLOCK-END
