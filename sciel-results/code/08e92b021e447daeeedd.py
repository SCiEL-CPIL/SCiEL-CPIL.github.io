import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Global Search
        # Since the minimum is in [-4, 4]^5, we bias our search there.
        # We use a mix of random points in [-4, 4] and [-5, 5] to be safe.
        # We will use 40% of the budget for this phase.
        n_global = int(budget * 0.4)
        
        # Generate initial points
        # We can use a simple random search.
        # To improve efficiency, we can evaluate points in batches.
        
        # Let's try a slightly structured approach:
        # 1. Center point (0,0,0,0,0)
        # 2. Random points in [-4, 4]^5
        # 3. Random points in [-5, 5]^5
        
        # First, evaluate center
        if calls < n_global and time.time() < deadline:
            evaluate(np.zeros(d))
            
        # Then, random points
        # We will generate points in chunks to allow for time checking
        chunk_size = 10
        
        while calls < n_global and time.time() < deadline:
            # Determine how many points to generate in this chunk
            n_pts = min(chunk_size, n_global - calls)
            if n_pts <= 0:
                break
            
            # Mix of points
            # 80% in [-4, 4], 20% in [-5, 5]
            n_inner = int(n_pts * 0.8)
            n_outer = n_pts - n_inner
            
            if n_inner > 0:
                X_inner = np.random.uniform(-4.0, 4.0, size=(n_inner, d))
                for x in X_inner:
                    if calls >= n_global or time.time() > deadline:
                        break
                    evaluate(x)
                    
            if n_outer > 0:
                X_outer = np.random.uniform(-5.0, 5.0, size=(n_outer, d))
                for x in X_outer:
                    if calls >= n_global or time.time() > deadline:
                        break
                    evaluate(x)
                    
        # Phase 2: Local Refinement
        # Use the best point found in Phase 1
        # We will use a simple gradient-free local search: Nelder-Mead
        # or a coordinate descent / pattern search.
        # Since scipy is available, let's use scipy.optimize.minimize with Nelder-Mead.
        # But we need to control the number of function calls strictly.
        
        # Alternative: Implement a simple local search ourselves to have full control.
        # Let's try a simple coordinate descent or a small-step random walk around the best point.
        # Or, use the remaining budget for a local search.
        
        n_local = budget - calls
        if n_local > 0 and best_x is not None:
            # We can try a few restarts of local search from the best points found.
            # For simplicity, let's just do one local search from the best point.
            # We'll use a simple local search: try moving in each dimension by small steps.
            
            # Let's implement a simple local search:
            # 1. Start from best_x
            # 2. Try moving in each dimension by +/- step_size
            # 3. If improvement, accept and continue in that direction
            # 4. If no improvement, reduce step_size
            # 5. Repeat until budget or time runs out
            
            current_x = best_x.copy()
            current_f = best_f
            step_size = 0.5  # Initial step size
            
            while calls < budget and time.time() < deadline:
                improved = False
                for i in range(d):
                    for direction in [-1, 1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        new_x = current_x.copy()
                        new_x[i] += direction * step_size
                        new_f = evaluate(new_x)
                        if new_f < current_f:
                            current_x = new_x
                            current_f = new_f
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break
                        
        return best_x, best_f
