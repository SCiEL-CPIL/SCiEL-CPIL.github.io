import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0  # Safety margin

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        best_x = np.zeros(n)
        best_fx = float('inf')
        
        # Fewer, better starting points to conserve budget
        np.random.seed(42)
        start_points = [np.zeros(n)]
        for _ in range(4):
            start_points.append(np.random.uniform(-4, 4, size=n))
        
        # Multi-start Powell with more iterations per start
        for i, x0 in enumerate(start_points):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            
            try:
                remaining = self.budget - self.calls
                # Allocate budget across remaining starts
                starts_left = len(start_points) - i
                max_iter = max(1, (remaining // (n + 1)) // max(1, starts_left) - 1)
                max_iter = min(max_iter, 80)
                
                res = minimize(eval_func, x0, method='Powell', 
                              options={'maxiter': max_iter, 'ftol': 1e-12, 'xtol': 1e-12})
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Final refinement from the best point
        remaining = self.budget - self.calls
        if remaining > 10 and best_x is not None and time.time() - self.start_time < self.time_limit:
            try:
                max_iter = min(30, remaining // (n + 1))
                if max_iter >= 1:
                    res = minimize(eval_func, best_x, method='Powell',
                                  options={'maxiter': max_iter, 'ftol': 1e-14, 'xtol': 1e-14})
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
