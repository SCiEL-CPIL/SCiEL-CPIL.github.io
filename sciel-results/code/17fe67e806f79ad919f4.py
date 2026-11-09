import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Lattice Search (3^5 = 243 points)
        # Grid points: -5, 0, 5
        grid_vals = [-5.0, 0.0, 5.0]
        
        # Generate all combinations
        import itertools
        grid_points = []
        for combo in itertools.product(grid_vals, repeat=n):
            grid_points.append(np.array(combo))
            
        n_grid = len(grid_points) # 243
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Evaluate grid points
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1
            
        # Phase 2: (1+1)-ES Refinement
        # Remaining budget
        remaining_budget = budget - calls_used
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Initialize ES parameters
        x = best_x.copy()
        f = best_f
        sigma = 1.0 # Initial step size
        
        rng = np.random.default_rng(42)
        
        # (1+1)-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Generate mutation
            z = rng.standard_normal(n)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            f_new = func(x_new)
            calls_used += 1
            
            # Update
            if f_new < f:
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                # Increase step size
                sigma *= 1.25
            else:
                # Decrease step size
                sigma *= 0.8
                
            # Optional: if sigma becomes too small, we might stop or restart?
            # For now, just continue. The step size adaptation handles convergence.
            
        return best_x, best_f
