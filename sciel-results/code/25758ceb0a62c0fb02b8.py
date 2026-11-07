import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = None
        
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Global Exploration
        # Sample 50 points to find promising basins
        n_explore = min(50, budget)
        explore_points = []
        explore_vals = []
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            if f is not None:
                explore_points.append(x)
                explore_vals.append(f)
        
        if best_x is None:
            return best_x, best_f

        # Phase 2: Multi-start (1+1)-ES with per-dimension step sizes
        # Select top 5 points from exploration
        if len(explore_points) > 0:
            # Sort by fitness
            sorted_indices = np.argsort(explore_vals)
            top_k = min(5, len(sorted_indices))
            start_points = [explore_points[i] for i in sorted_indices[:top_k]]
        else:
            start_points = [best_x]

        # Allocate remaining budget to these start points
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f
            
        n_starts = len(start_points)
        evals_per_start = max(1, remaining_budget // n_starts)
        
        # (1+1)-ES with per-dimension sigma
        def run_es(x_start, n_evals):
            nonlocal calls, best_f, best_x
            
            # Evaluate start point if not already done (it was done in exploration, 
            # but we need the current value for the local loop logic)
            # We can reuse the value from explore_vals if we tracked it, 
            # but it's safer to just use the global best_f for comparison 
            # or re-evaluate if we didn't keep the mapping. 
            # Since we kept explore_vals, let's just assume x_start is already evaluated 
            # and use its value. However, for simplicity and robustness against 
            # state loss, we will just proceed with the ES loop.
            
            # Get current fitness of x_start. 
            # We know it was evaluated in Phase 1. 
            # We can look it up, but it's easier to just let the ES loop 
            # use the global best_f for comparison if we don't have the local f.
            # Actually, let's just re-evaluate if necessary? No, that wastes budget.
            # Let's assume x_start has fitness f_start. 
            # We can find it from explore_vals.
            
            # Find the index of x_start in explore_points to get its value
            # This is inefficient but n is small.
            f_current = None
            for i, p in enumerate(explore_points):
                if np.allclose(p, x_start):
                    f_current = explore_vals[i]
                    break
            
            if f_current is None:
                f_current = evaluate(x_start)
                if f_current is None:
                    return x_start, best_f
            
            x = x_start.copy()
            f = f_current
            
            # Per-dimension step sizes
            sigma = np.ones(dim) * 1.5
            
            for i in range(n_evals):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate mutant
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = evaluate(x_trial)
                if f_trial is None:
                    break
                
                # Update best if improved
                if f_trial < f:
                    f = f_trial
                    x = x_trial
                    # Increase sigma
                    sigma *= 1.2
                else:
                    # Decrease sigma
                    sigma *= 0.8
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            return x, f

        # Run ES from each start point
        for sp in start_points:
            if calls >= budget or time.time() > deadline:
                break
            run_es(sp, evals_per_start)
            
        return best_x, best_f
