"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 300s limit
        
        call_count = 0
        def counting_func(x):
            nonlocal call_count
            if time.time() > deadline:
                return 1e10
            call_count += 1
            return func(x)
            
        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * self.dim
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: 10 starts, 20 fun evals each
        n_starts_1 = 10
        maxfun_1 = 20
        candidates = []
        
        for i in range(n_starts_1):
            if call_count >= self.budget or time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            # maxiter is a safety cap, maxfun is the primary budget controller
            # L-BFGS-B in 5D uses ~6-8 function calls per iteration
            res = minimize(counting_func, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 10, 'maxfun': maxfun_1})
            
            if np.isfinite(res.fun) and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
            
            candidates.append((res.fun, res.x.copy()))
            
        # Phase 2: Top 5, 30 fun evals each
        if call_count < self.budget and time.time() < deadline:
            candidates.sort(key=lambda x: x[0])
            top_k = min(5, len(candidates))
            for i in range(top_k):
                if call_count >= self.budget or time.time() > deadline:
                    break
                x0 = candidates[i][1]
                res = minimize(counting_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 10, 'maxfun': 30})
                
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                
                candidates[i] = (res.fun, res.x.copy())

        # Phase 3: Top 2, 30 fun evals each
        if call_count < self.budget and time.time() < deadline:
            candidates.sort(key=lambda x: x[0])
            top_k = min(2, len(candidates))
            for i in range(top_k):
                if call_count >= self.budget or time.time() > deadline:
                    break
                x0 = candidates[i][1]
                res = minimize(counting_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 10, 'maxfun': 30})
                
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                
                candidates[i] = (res.fun, res.x.copy())

        # Phase 4: Top 1, remaining budget
        if call_count < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - call_count
            if remaining > 0:
                res = minimize(counting_func, best_x, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'maxfun': remaining})
                
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
