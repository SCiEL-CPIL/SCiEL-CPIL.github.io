import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 280s safety margin out of 300s limit

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Reduced initial sampling - just enough to find a good starting region
        n_init = min(30, self.budget // 10)
        init_points = []
        init_vals = []
        
        for _ in range(n_init):
            if time.time() > deadline or self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = safe_func(x)
            init_points.append(x)
            init_vals.append(f)
        
        if not init_points:
            x = np.zeros(self.dim)
            safe_func(x)
            return self.best_x, self.best_f

        # Find the best initial point
        best_idx = np.argmin(init_vals)
        x0 = init_points[best_idx].copy()
        
        # Phase 2: CMA-ES style adaptive search using scipy's Nelder-Mead as a fallback
        # First, try a few local refinements with different step sizes
        local_steps = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        current_x = x0.copy()
        current_f = self.best_f
        
        for step in local_steps:
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Coordinate descent with adaptive steps
            for d in range(self.dim):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                for direction in [1, -1]:
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    new_x = current_x.copy()
                    new_x[d] += direction * step
                    new_x = np.clip(new_x, -5, 5)
                    
                    f_new = safe_func(new_x)
                    
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
        
        # Phase 3: If we have budget left, use Nelder-Mead for fine-tuning
        if self.calls < self.budget - 50 and time.time() < deadline:
            try:
                # Use scipy's Nelder-Mead with a tight budget
                nm_budget = min(200, self.budget - self.calls - 10)
                
                def nm_wrapper(x):
                    if self.calls >= self.budget or time.time() > deadline:
                        return self.best_f + 1000
                    return safe_func(np.clip(x, -5, 5))
                
                result = minimize(nm_wrapper, current_x, method='Nelder-Mead',
                                  options={'maxiter': nm_budget, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 4: Final random restarts if budget remains
        if self.calls < self.budget - 5:
            remaining_budget = self.budget - self.calls
            n_restarts = min(3, remaining_budget // 15)
            
            for r in range(n_restarts):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                x_r = rng.uniform(-5, 5, self.dim)
                f_r = safe_func(x_r)
                
                if f_r < self.best_f:
                    current_x = x_r.copy()
                    current_f = f_r
                else:
                    current_x = self.best_x.copy()
                    current_f = self.best_f
                
                # Quick local refinement
                for step in [0.5, 0.1, 0.01]:
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    for d in range(self.dim):
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            if self.calls >= self.budget or time.time() > deadline:
                                break
                            new_x = current_x.copy()
                            new_x[d] += direction * step
                            new_x = np.clip(new_x, -5, 5)
                            f_new = safe_func(new_x)
                            if f_new < current_f:
                                current_x = new_x
                                current_f = f_new
        
        return self.best_x, self.best_f
