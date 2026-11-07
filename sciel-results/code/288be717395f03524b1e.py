import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        import time
        self.deadline = time.time() + 290 # 10s margin
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = func(x)
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Strategy: Multi-start Nelder-Mead
        # Start 1: Origin
        # Start 2: Best from Start 1 + noise
        # Start 3: Random
        
        starts = [
            np.zeros(self.dim),
            # Add more starts if budget allows
        ]
        
        # We need to distribute budget.
        # Let's do 3 runs.
        # Run 1: 200 calls
        # Run 2: 150 calls
        # Run 3: 150 calls
        
        # Since we can't easily limit scipy's internal calls to exactly N, 
        # we use maxiter. 
        # For Nelder-Mead, 1 iteration ~ 6-10 calls.
        # 200 calls ~ 20-30 iterations.
        
        x0_list = [
            np.zeros(self.dim),
            np.random.uniform(-2, 2, self.dim),
            np.random.uniform(-4, 4, self.dim)
        ]
        
        budgets = [200, 150, 150]
        
        for i, x0 in enumerate(x0_list):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline - 5:
                break
                
            # Estimate maxiter
            # Roughly 8 calls per iteration for n=5
            maxiter = max(1, (budgets[i] - (self.calls % 10)) // 8)
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-8})
            except TimeoutError:
                break
            except Exception:
                continue
            
            # Update best
            if self.best_f > res.fun:
                self.best_f = res.fun
                self.best_x = res.x
        
        return self.best_x, self.best_f
