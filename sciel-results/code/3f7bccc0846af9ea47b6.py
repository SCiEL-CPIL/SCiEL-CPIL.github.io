import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # The evaluator kills after 330s. We have 300 calls.
        # Assuming ~1ms per call, 300 calls is 0.3s.
        # Time limit is not the primary constraint unless func is slow.
        # But we add a guard just in case.
        deadline = start_time + 290.0
        
        # 1. Initialization
        # Evaluate center and a few random points to find a good starting point
        # We allocate 10% of the budget to initialization
        init_budget = max(5, self.budget // 10)
        rng = np.random.default_rng(123)
        
        best_x = np.zeros(self.dim)
        best_f = func(best_x)
        
        # Add some random points
        for _ in range(init_budget - 1):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        calls_made = init_budget
        remaining = self.budget - calls_made
        
        # 2. Main Loop: (1+1)-ES with Restarts
        # We will run multiple phases of (1+1)-ES.
        # Each phase runs until stagnation or budget/time limit.
        
        # Parameters for (1+1)-ES
        # Sigma starts at 1.0
        # Success rule: if improved, sigma *= 1.5, else sigma *= 0.8
        # Stagnation limit: 10 * dim
        
        while remaining > 0:
            if time.time() > deadline:
                break
                
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            stagnation = 0
            stagnation_limit = 10 * self.dim
            
            # Run (1+1)-ES
            while remaining > 0:
                if time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    stagnation += 1
                    sigma *= 0.8
                    # If sigma becomes too small, we are stuck
                    if sigma < 1e-5:
                        break
                        
                if stagnation > stagnation_limit:
                    break
            
            # Restart Strategy
            if remaining > 0:
                # If we broke out due to stagnation or small sigma
                # Restart from a random point to escape local minima
                # This is crucial for Rastrigin/Gallagher
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                remaining -= 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                else:
                    # If the random point is worse, we might want to keep searching from the old best
                    # but with a larger sigma?
                    # Or just continue from the new random point?
                    # Continuing from the new random point forces exploration.
                    # Let's continue from the new random point.
                    pass 

        return best_x, best_f
