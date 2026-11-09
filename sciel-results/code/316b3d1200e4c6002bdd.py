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
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x)
            f_val = func(x_arr)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with (1+1)-ES and aggressive restarts
        # Use 60% of budget for global exploration
        global_budget = int(self.budget * 0.6)
        
        # Initial point
        x = rng.uniform(-5, 5, n)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 40  # More aggressive: explore every 40 calls
        
        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15  # Faster adaptation for better basin finding
                stagnation = 0
            else:
                sigma *= 0.85  # Faster contraction to escape deceptive regions
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-5, 5, n)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.0
                    stagnation = 0
                    # Reduce threshold for more aggressive restarts later
                    max_stagnation = max(15, max_stagnation // 2)

        # Phase 2: Local Refinement with L-BFGS-B then Nelder-Mead
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            
            # First: L-BFGS-B for efficient gradient-based refinement
            max_iter_lbfgs = max(10, remaining_calls // 3)
            try:
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': max_iter_lbfgs,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass
            
            # Second: Nelder-Mead polish if budget remains
            if calls < self.budget and time.time() < deadline:
                remaining_calls = self.budget - calls
                if remaining_calls > 10:
                    max_iter_nm = max(5, remaining_calls // (n + 2))
                    try:
                        result = scipy.optimize.minimize(
                            safe_func, 
                            best_x, 
                            method='Nelder-Mead', 
                            options={
                                'maxiter': max_iter_nm,
                                'xatol': 1e-6,
                                'fatol': 1e-6,
                                'adaptive': True
                            }
                        )
                    except (TimeoutError, StopIteration, Exception):
                        pass

        return best_x, best_f
