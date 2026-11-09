import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Lattice Search (3^5 = 243 points)
        grid_vals = [-5.0, 0.0, 5.0]
        grid_points = [np.array(combo) for combo in itertools.product(grid_vals, repeat=n)]
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        results = [] # List of (f, x) for all grid points
        
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            fx = func(x)
            calls_used += 1
            results.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1
            results.append((best_f, best_x))

        # Phase 2: Multi-start Local Search from Top K Grid Points
        # Sort results by fitness
        results.sort(key=lambda item: item[0])
        
        # We have used ~243 calls. Remaining budget ~257.
        # Let's pick top 10 points.
        # Allocate remaining budget among them.
        k_start = min(10, len(results))
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Budget per start point
        budget_per_start = max(1, remaining_budget // k_start)
        
        rng = np.random.default_rng(42)
        
        for i in range(k_start):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Start from the i-th best grid point
            f_curr = results[i][0]
            x_curr = results[i][1].copy()
            
            # Local Search: (1+1)-ES
            sigma = 0.5 # Smaller initial step for local refinement
            # Adapt sigma based on dimension or just fixed small steps?
            # Let's use a simple adaptive step.
            
            local_calls = 0
            max_local_calls = budget_per_start
            
            while local_calls < max_local_calls and calls_used < budget:
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(n)
                x_new = x_curr + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_used += 1
                local_calls += 1
                
                if f_new < f_curr:
                    x_curr = x_new
                    f_curr = f_new
                    # Slow down step size on success
                    sigma *= 0.9
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                else:
                    # Increase step size on failure to explore
                    sigma *= 1.1
                    # Prevent sigma from exploding
                    if sigma > 2.0:
                        sigma = 2.0
                    if sigma < 0.01:
                        sigma = 0.01
                        
        return best_x, best_f
