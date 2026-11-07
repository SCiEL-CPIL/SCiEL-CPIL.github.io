import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            return f_val

        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Multi-Start Powell
        num_restarts = 5
        calls_per_restart = self.budget // num_restarts
        
        for i in range(num_restarts):
            if self.call_count >= self.budget or time.time() > deadline:
                break
            
            # Random start
            x0 = np.random.uniform(-5.0, 5.0, self.dim)
            
            try:
                res = minimize(wrapped_func, x0, method='Powell', bounds=bounds, 
                               options={'maxfev': calls_per_restart, 'maxiter': 50})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                pass
        
        # Final Polish with Nelder-Mead if budget remains
        if self.call_count < self.budget and self.best_x is not None and time.time() < deadline:
            remaining = self.budget - self.call_count
            try:
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead', 
                               options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                pass

        return self.best_x, self.best_f
