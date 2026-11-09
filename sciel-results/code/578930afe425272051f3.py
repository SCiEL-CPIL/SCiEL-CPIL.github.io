import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # 1. Global Exploration with Sobol
        # Generate 10 Sobol points
        # Use a simple Sobol generator or just random if Sobol is complex to implement quickly?
        # NumPy has no built-in Sobol. We can use a simple stratified sampling or just random.
        # Stratified sampling is easy: divide [-5, 5] into 2^dim bins? No, too many.
        # Let's use random but with a fixed seed for reproducibility in logic, 
        # but the task says "hidden seeded shift", so our seed doesn't matter for the function, 
        # only for our consistency.
        
        # Let's use a simple Halton sequence or just random.
        # For 5D, random is fine if we do 10-20 points.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: 10 Global Random Points
        n_global = 10
        points = rng.uniform(-5, 5, (n_global, self.dim))
        values = np.array([func(p) for p in points])
        
        best_idx = np.argmin(values)
        x_best = points[best_idx].copy()
        f_best = values[best_idx]
        
        calls_used = n_global
        
        # Phase 2: Local Polish on each global point
        n_polish = 20
        # We have budget - n_global left.
        # We want to polish all 10 points.
        # Cost per polish: n_polish calls.
        # Total cost: 10 * 20 = 200.
        # Remaining: 500 - 10 - 200 = 290.
        
        for i in range(n_global):
            if calls_used >= self.budget: break
            if time.time() > deadline: break
            
            x_curr = points[i].copy()
            f_curr = values[i]
            
            # Simple (1+1)-ES for n_polish steps
            sigma = 1.0
            for _ in range(n_polish):
                if calls_used >= self.budget: break
                if time.time() > deadline: break
                
                y = np.clip(x_curr + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    
            if f_curr < f_best:
                x_best, f_best = x_curr, f_curr
        
        # Phase 3: Final Refinement
        # Use remaining budget for a high-precision search from x_best
        remaining = self.budget - calls_used
        if remaining > 0:
             # Use Nelder-Mead or a tighter (1+1)-ES
             # Let's use Nelder-Mead from scipy as it's robust for final convergence
             from scipy.optimize import minimize
             
             # We need to wrap func to count calls and respect deadline
             # But minimize doesn't easily let us stop on deadline.
             # Let's implement a tight (1+1)-ES for the remainder.
             sigma = 0.5
             for _ in range(remaining):
                 if time.time() > deadline: break
                 
                 y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
                 fy = func(y)
                 calls_used += 1
                 
                 if fy < f_best:
                     x_best, f_best = y, fy
                     sigma *= 1.1
                 else:
                     sigma *= 0.9
                     
        return x_best, f_best
