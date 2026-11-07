import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        
        def wrapper(x):
            nonlocal calls
            if time.time() > deadline:
                return 1e10
            calls += 1
            return func(x)
            
        # 1. Global-ish search with Nelder-Mead
        x_best = rng.uniform(-5, 5, d)
        f_best = wrapper(x_best)
        
        # Run a few Nelder-Meads
        for i in range(3):
            if calls > budget - 50: # Leave budget for polish
                break
            x0 = x_best + rng.normal(0, 0.5, d)
            x0 = np.clip(x0, -5, 5)
            # Simple manual Nelder-Mead to control calls strictly? 
            # Or just use scipy with maxiter
            # Let's use a simple random search + local refinement to avoid scipy overhead/bugs
            pass
            
        # Let's implement a simple (1+1) CMA-like or just Nelder-Mead manually for control
        # Actually, the previous manual Nelder-Mead was fine. Let's use that.
        
        # ... (Insert Manual Nelder-Mead Code) ...
        
        # 2. Polish with Coordinate Descent
        # Use remaining budget
        while calls < budget and time.time() < deadline:
            improved = False
            for i in range(d):
                if calls >= budget: break
                # Try small steps
                for step in [0.1, 0.01, 0.001]:
                    x_try = x_best.copy()
                    x_try[i] += step
                    if x_try[i] > 5: continue
                    f_try = wrapper(x_try)
                    if f_try < f_best:
                        x_best = x_try
                        f_best = f_try
                        improved = True
                    x_try = x_best.copy()
                    x_try[i] -= step
                    if x_try[i] < -5: continue
                    f_try = wrapper(x_try)
                    if f_try < f_best:
                        x_best = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                break
                
        return x_best, f_best
