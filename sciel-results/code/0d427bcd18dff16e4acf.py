import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                # Raise exception to stop scipy? 
                # Better to just return a large value and let it stop?
                # Or check time in the loop outside?
                # Scipy doesn't easily allow breaking out of the loop from inside the function.
                # We must rely on maxfev and the fact that it's fast.
                # But if it hangs, we fail.
                # 500 calls is very fast. Time limit is 300s.
                # So time is not a real constraint for 500 calls of a simple function.
                pass
            
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Multi-start Nelder-Mead
        
        # Phase 1: 4 restarts, 50 calls each
        # Phase 2: 1 final polish, 300 calls
        
        rng = np.random.default_rng(42)
        
        # We need to be careful with the budget.
        # 4 * 50 = 200.
        # Remaining 300.
        
        restarts = 4
        calls_per_restart = 50
        final_calls = budget - (restarts * calls_per_restart)
        
        candidates = []
        
        for i in range(restarts):
            if time.time() > deadline:
                break
                
            # Random start
            x0 = rng.uniform(-5, 5, n)
            
            # Initial simplex: x0 + 0.05 * range? 
            # Nelder-Mead default is x0 + 0.05 if x0[i]!=0 else 0.00025
            # Let's specify a simplex with a reasonable size, e.g., 0.5
            # But if the valley is narrow, 0.5 might be too big?
            # Actually, NM adapts.
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxfev': calls_per_restart, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                candidates.append((res.fun, res.x))
            except Exception:
                pass
                
        # Find best candidate
        if candidates:
            best_candidate_f, best_candidate_x = min(candidates, key=lambda x: x[0])
        else:
            # Fallback if all failed
            best_candidate_f = best_f
            best_candidate_x = best_x
            
        # Phase 2: Final Polish
        if calls_made < budget:
            remaining = budget - calls_made
            # Ensure we don't exceed budget
            # Note: wrapped_func increments calls_made.
            # We set maxfev to remaining.
            try:
                res_final = minimize(
                    wrapped_func, 
                    best_candidate_x, 
                    method='Nelder-Mead',
                    options={'maxfev': remaining, 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except Exception:
                pass
                
        return best_x, best_f
