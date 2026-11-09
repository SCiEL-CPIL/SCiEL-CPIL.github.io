import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
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

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 60% of budget for global exploration
        global_budget = int(self.budget * 0.6)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 50
        
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
                sigma *= 1.2
                stagnation = 0
            else:
                sigma *= 0.9
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-5, 5, self.dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.0
                    stagnation = 0
                    max_stagnation = max(15, max_stagnation // 2) # Reduce threshold for more aggressive restarts

        # Phase 2: Local Refinement
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            
            # Use L-BFGS-B for local refinement as it's more efficient for smooth functions
            # L-BFGS-B uses gradient approximations and can converge faster
            max_iter = max(10, remaining_calls // 2)
            
            try:
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass
                
            # If we still have budget, do a final Nelder-Mead polish
            if calls < self.budget and time.time() < deadline:
                remaining_calls = self.budget - calls
                if remaining_calls > 10:
                    max_iter = max(5, remaining_calls // (self.dim + 2))
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

        return best_x, best_f
