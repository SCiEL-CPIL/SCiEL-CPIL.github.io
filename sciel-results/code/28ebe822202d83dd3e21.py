import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(d)
        best_f = float('inf')
        total_calls = 0
        
        # Strategy: Combine Multi-Start (1+1)-ES with 1/5 Success Rule
        # The (1+1)-ES is robust and simple. For ill-conditioned problems,
        # a global step size might be suboptimal, but multi-start helps
        # escape local minima. We will use a slightly more aggressive
        # step size adaptation and multiple restarts.
        
        num_restarts = 8
        calls_per_restart = budget // num_restarts
        
        # Initial points: center and random points
        starts = [np.zeros(d)]
        for _ in range(num_restarts - 1):
            x0 = rng.uniform(-4, 4, size=d)
            starts.append(x0)
        
        c_inc = 1.1
        c_dec = 0.9
        
        for i, x0 in enumerate(starts):
            if total_calls >= budget:
                break
            
            remaining = budget - total_calls
            if i == len(starts) - 1:
                max_calls = remaining
            else:
                max_calls = min(calls_per_restart, remaining)
            
            if max_calls <= 1:
                break
            
            # (1+1)-ES
            x = np.array(x0, dtype=float)
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.5
            
            while total_calls < max_calls:
                if time.time() > deadline - 1.0:
                    break
                    
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= c_inc
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= c_dec
                
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                if time.time() > deadline - 1.0:
                    break

        return best_x, best_f
