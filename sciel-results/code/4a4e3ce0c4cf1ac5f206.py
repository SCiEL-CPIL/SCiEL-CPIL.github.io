import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 70% of budget for global exploration to allow better convergence
        global_budget = int(self.budget * 0.7)
        
        # Initial point: start at center for faster initial convergence
        x = np.zeros(self.dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 40
        restart_count = 0
        max_restarts = 10
        
        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                stagnation = 0
            else:
                sigma *= 0.85
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation and restart_count < max_restarts:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-5, 5, self.dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.5
                    stagnation = 0
                    max_stagnation = max(15, max_stagnation // 2)
                    restart_count += 1

        # Phase 2: Local Refinement with Nelder-Mead
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            # Estimate max iterations for Nelder-Mead
            # Each iteration uses ~dim function evaluations
            max_iter = max(10, remaining_calls // (self.dim + 2))
            
            try:
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass

        # Phase 3: Final high-precision polish pass
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 10:
                try:
                    result = scipy.optimize.minimize(
                        safe_func, 
                        best_x, 
                        method='Powell', 
                        options={
                            'maxiter': max(5, remaining_calls // 10),
                            'ftol': 1e-8,
                            'xtol': 1e-8
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

        return best_x, best_f
