import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin

        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.full(n, np.nan)
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Coarse Global Search ---
        # Use a mix of random and LHS-like sampling to cover the space
        # We'll do about 30% of the budget here
        global_budget = int(total_budget * 0.3)
        
        # Simple LHS sampling
        # For 5D, we can divide each dimension into k bins
        # Let's pick k=5 for a 5^5 = 3125 grid? No, budget is 500.
        # Let's just do random sampling with a "halo" around the origin and corners.
        
        # 1. Try corners and center
        points = []
        # Center
        points.append(np.zeros(n))
        # Axes extremes
        for i in range(n):
            for val in [-5, 5]:
                p = np.zeros(n)
                p[i] = val
                points.append(p)
        
        # Add some random points
        # We want to fill the rest of the global_budget
        num_random = global_budget - len(points)
        if num_random > 0:
            # Use a fixed seed for reproducibility of the structure, but the function is non-deterministic?
            # The task says "hidden seeded shift", so the function is fixed per instance.
            # Using a fixed rng here is fine as it's just sampling.
            rng = np.random.default_rng(123)
            random_points = rng.uniform(-5, 5, (num_random, n))
            points.extend(random_points)
            
        # Evaluate all global points
        for p in points:
            if not time_ok() or calls_made >= global_budget:
                break
            do_call(p)

        # --- Phase 2: Local Refinement ---
        # Use the best point found so far as the starting point for local optimization.
        # We can also try a few other good points if we have budget.
        
        remaining_budget = total_budget - calls_made
        
        # Sort points by fitness? We didn't store them all.
        # Let's just use the current best_x.
        
        if remaining_budget > 10 and best_x is not None and not np.any(np.isnan(best_x)):
            # Use Nelder-Mead
            # We need to wrap the function to respect the budget and time
            nm_calls = 0
            
            def nm_func(x):
                nonlocal nm_calls
                if nm_calls >= remaining_budget:
                    return best_f
                if not time_ok():
                    return best_f
                nm_calls += 1
                # We need to count this call in the main counter?
                # The do_call function increments calls_made.
                # So we should use do_call inside nm_func?
                # But do_call updates best_f and best_x.
                # NM needs to see the raw values to converge.
                # If we update best_f inside, it's fine.
                
                # However, NM might evaluate the same point multiple times?
                # It's safe to just call func directly and manage the counter.
                
                val = do_call(x)
                return val

            try:
                # maxiter should be set high enough to use the budget
                # NM uses ~1-2 calls per iteration
                max_iter = int(remaining_budget * 0.8)
                
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # If we still have budget, do a simple coordinate descent or another NM restart from best_x with a smaller simplex?
        # Or just let NM run if it stopped early due to maxiter but budget remains.
        
        remaining_budget = total_budget - calls_made
        if remaining_budget > 5 and best_x is not None and not np.any(np.isnan(best_x)) and time_ok():
             # Try a smaller step size NM
            nm_calls2 = 0
            def nm_func2(x):
                nonlocal nm_calls2
                if nm_calls2 >= remaining_budget:
                    return best_f
                if not time_ok():
                    return best_f
                nm_calls2 += 1
                val = do_call(x)
                return val
            
            try:
                # Shrink the initial simplex by using a smaller xatol/fatol?
                # Or just run again. NM is robust.
                max_iter2 = int(remaining_budget * 0.8)
                res2 = minimize(nm_func2, best_x, method='Nelder-Mead', 
                                options={'maxiter': max_iter2, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        return best_x, best_f
