import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        dim = self.dim
        budget = self.budget
        
        # Multi-start HJ
        # Split budget among starts
        n_starts = 5
        budget_per_start = budget // n_starts
        
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(0)
        
        for s in range(n_starts):
            if time.time() > deadline:
                break
                
            # Initial point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # HJ search
            # Initial step size
            delta = 0.5
            min_delta = 1e-5
            
            # HJ loop
            # We use a local budget for this start
            local_budget = budget_per_start
            # 1 eval for init already done? No, we counted it in best_f update?
            # Let's say we have local_budget left for HJ steps.
            # Each HJ step involves 2 evals per coordinate (worst case) or 1.
            # Let's limit the number of HJ iterations.
            max_iter = local_budget // (2 * dim) # Rough estimate
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                improved = False
                step = 0.0
                for i in range(dim):
                    # Try +
                    y = x.copy()
                    y[i] += delta
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        step[i] = delta
                        improved = True
                        delta *= 2.0
                    else:
                        # Try -
                        y = x.copy()
                        y[i] -= delta
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            step[i] = -delta
                            improved = True
                            delta *= 2.0
                        else:
                            delta *= 0.5
                            # If delta is too small, reset to initial for this coord?
                            # Standard HJ reduces delta.
                            if delta < min_delta:
                                delta = 0.5 # Reset
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        
                # Pattern move
                if improved:
                    y = x + step
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Pattern move failed, reduce delta
                        delta *= 0.5
                        
        return best_x, best_f
