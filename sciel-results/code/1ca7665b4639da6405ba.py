import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290  # 10s safety margin
        
        def wrapped_func(x):
            self.eval_count += 1
            if self.eval_count > self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return float('inf')
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search
        n_global = 100
        if self.budget < 200:
            n_global = self.budget // 2
        else:
            n_global = min(100, self.budget - 100)

        candidates = []
        halton_points = self._halton_sequence(n_global, self.dim)
        
        for p in halton_points:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            fx = wrapped_func(p)
            candidates.append((fx, p))
            
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Search
        k_local = 3
        remaining_budget = self.budget - self.eval_count
        local_budget_per = remaining_budget // k_local if k_local > 0 else 0
        
        if local_budget_per > 20:
            max_iter = local_budget_per // 6
            for i in range(min(k_local, len(candidates))):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                if remaining_budget - (self.eval_count - (self.budget - remaining_budget)) < 20:
                     # This check is messy, just check remaining
                     pass
                
                # Check remaining budget dynamically
                rem = self.budget - self.eval_count
                if rem < 20:
                    break
                    
                start_x = candidates[i][1]
                
                try:
                    # Use maxiter to limit iterations, but wrapped_func limits evals
                    res = minimize(wrapped_func, start_x, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                except Exception:
                    pass
                    
        return self.best_x, self.best_f

    def _halton_sequence(self, count, dim):
        points = []
        for i in range(1, count + 1):
            point = []
            for d in range(1, dim + 1):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= d
                    r += f * (i_val % d)
                    i_val //= d
                point.append(r * 10 - 5)
            points.append(np.array(point))
        return points
