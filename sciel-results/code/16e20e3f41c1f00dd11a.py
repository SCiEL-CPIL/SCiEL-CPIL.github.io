import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Multi-Start Nelder-Mead
        n_starts = 10
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # We need to track calls to ensure we don't exceed budget
        # scipy minimize doesn't easily expose a strict call counter for early stopping 
        # unless we wrap the function.
        
        class CallCounter:
            def __init__(self, func, budget_limit):
                self.func = func
                self.limit = budget_limit
                self.count = 0
                self.exceeded = False
            
            def __call__(self, x):
                if self.count >= self.limit:
                    self.exceeded = True
                    # Return a bad value to stop optimization
                    return 1e10
                self.count += 1
                return self.func(x)
        
        for i in range(n_starts):
            if calls_made >= budget:
                break
            
            # Determine starting point
            if i == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Mix of random and perturbed best
                if rng.random() < 0.5 and best_x is not None:
                    # Perturb best found so far
                    scale = 1.0 / (1 + i) # Reduce perturbation over time
                    x0 = best_x + rng.normal(0, scale, dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
            
            # Allocate remaining budget to this start
            remaining = budget - calls_made
            # Give at least 10 calls, but not more than remaining
            alloc = min(remaining, max(10, budget // n_starts))
            
            counter = CallCounter(func, alloc)
            
            try:
                res = minimize(counter, x0, method='Nelder-Mead', 
                               options={'maxiter': alloc, 'xatol': 1e-6, 'fatol': 1e-6})
            except Exception:
                pass
            
            calls_made += counter.count
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        # If we haven't used all budget, do a final local polish on best_x
        if best_x is not None and calls_made < budget:
             counter2 = CallCounter(func, budget - calls_made)
             try:
                 res2 = minimize(counter2, best_x, method='Nelder-Mead', 
                                options={'maxiter': budget - calls_made, 'xatol': 1e-8, 'fatol': 1e-8})
                 if res2.fun < best_f:
                     best_f = res2.fun
                     best_x = res2.x
             except Exception:
                 pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
