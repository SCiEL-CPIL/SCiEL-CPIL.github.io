import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES with 1/5 success rule
        # This is fast and effective for the initial phase of the Sphere function
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        count = 1
        
        # Run (1+1)-ES for a portion of the budget
        # The 1/5 rule suggests increasing sigma by 1.5 on success and 0.87 on failure
        # We use this for the first 200 evaluations
        phase1_budget = min(200, budget - 1)
        
        for _ in range(phase1_budget):
            if time.time() > deadline:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            count += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
                
            # If sigma becomes too small, we might be converged
            if sigma < 1e-6:
                break

        # Phase 2: If we have budget left, do a local refinement
        # Since the Sphere function is convex and smooth, we can use a simple gradient-free
        # method like a fine-grained random search around the best point found
        
        if count < budget:
            # Try a few more random steps with decreasing sigma
            sigma = max(sigma, 0.1)
            for _ in range(budget - count):
                if time.time() > deadline:
                    break
                    
                # Decrease sigma gradually
                sigma *= 0.99
                
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
