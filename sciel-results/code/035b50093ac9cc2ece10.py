import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        start_time = time.time()
        max_time = 290.0
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= total_budget:
                return np.inf
            calls_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Iterated Coordinate Descent
        # 5 restarts, 100 calls each
        n_restarts = 5
        calls_per_restart = total_budget // n_restarts # 100
        
        for r in range(n_restarts):
            if calls_used >= total_budget:
                break
            if time.time() - start_time > max_time:
                break
            
            # Initialize
            if r == 0:
                x = rng.uniform(-5, 5, n)
            else:
                # Perturb best
                # Larger perturbation to ensure we leave the basin
                x = best_x + rng.standard_normal(n) * 3.0
                x = np.clip(x, -5, 5)
            
            fx = evaluate(x)
            
            # Coordinate Descent
            step_size = 2.0
            min_step = 0.005
            # Max passes limited by budget
            # Each pass costs n * 2 calls (worst case)
            # We have calls_per_restart - 1 calls left
            max_passes = (calls_per_restart - 1) // (2 * n) # 100-1 // 10 = 9 passes
            
            for p in range(max_passes):
                if calls_used >= total_budget:
                    break
                if time.time() - start_time > max_time:
                    break
                
                improved = False
                for i in range(n):
                    if calls_used >= total_budget:
                        break
                    if time.time() - start_time > max_time:
                        break
                    
                    # Try +
                    y = x.copy()
                    y[i] += step_size
                    y = np.clip(y, -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
                    else:
                        # Try -
                        y = x.copy()
                        y[i] -= step_size
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                    
                if not improved:
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                # If improved, keep step_size to continue moving in that direction
                # This helps cross flat regions if the step is large enough
        
        return best_x, best_f
