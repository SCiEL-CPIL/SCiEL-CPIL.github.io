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
            
        # Phase 2: (1+1)-ES Refinement with better step-size adaptation
        # Remaining budget
        remaining_budget = budget - calls_used
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Initialize ES parameters
        x = best_x.copy()
        f = best_f
        sigma = 1.0 # Initial step size
        
        rng = np.random.default_rng(42)
        
        # (1+1)-ES loop with 1/5th rule step-size adaptation
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
                # Increase step size (1/5th rule: increase when success)
                sigma *= 1.3
            else:
                # Decrease step size
                sigma *= 0.75
            
            # Floor on sigma to avoid numerical issues
            if sigma < 1e-12:
                sigma = 1e-12
        
        # Phase 3: Final high-precision polish pass
        # Use a smaller, more focused search around the best point
        polish_budget = min(20, budget - calls_used)
        if polish_budget > 0 and time.time() < deadline:
            # Do a local Nelder-Mead-like refinement or fine grid
            # Use a small grid around the best point
            polish_sigma = max(sigma * 0.5, 1e-4)
            for i in range(polish_budget):
                if calls_used >= budget or time.time() > deadline:
                    break
                # Try small perturbations in each dimension
                for d in range(n):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    x_try = x.copy()
                    x_try[d] += polish_sigma
                    x_try[d] = np.clip(x_try[d], -5, 5)
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < f:
                        x = x_try
                        f = f_try
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    x_try = x.copy()
                    x_try[d] -= polish_sigma * 2
                    x_try[d] = np.clip(x_try[d], -5, 5)
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < f:
                        x = x_try
                        f = f_try
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    polish_sigma *= 0.7
        
        return best_x, best_f
