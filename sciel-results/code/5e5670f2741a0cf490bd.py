# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer using Nelder-Mead for fast local convergence on f7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return float('inf')
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Global exploration with multiple Nelder-Mead starts
        # f7 is step ellipsoidal, so local minima are common. 
        # We need to find the global basin.
        
        # Generate diverse starting points
        num_starts = 15
        start_points = []
        start_points.append(np.zeros(dim))
        for i in range(num_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, size=dim))

        # Sort starting points by initial function value? 
        # Let's just run them in order, but maybe prioritize center first.
        
        # Allocate budget: 
        # We want to give each start a decent chance, but not waste too much if it fails.
        # Nelder-Mead is fast. Let's give it up to 50 iterations per start.
        # If it gets stuck, we move to the next.
        
        # Dynamic budget allocation:
        # Start with a small budget per start, increase if successful.
        
        # Let's try a strategy:
        # 1. Evaluate all start points (num_starts calls)
        # 2. Pick the best one and run NM with a larger budget.
        # 3. If that converges well, maybe run NM on the second best, etc.
        # 4. Finally, polish with (1+1)-ES.

        # Step 1: Evaluate start points
        start_vals = []
        for sp in start_points:
            if calls >= budget or time.time() > deadline:
                break
            v = wrapper(sp)
            start_vals.append((v, sp))
        
        if not start_vals:
            return best_x, best_f

        # Sort by value
        start_vals.sort(key=lambda x: x[0])
        
        # Step 2: Run Nelder-Mead on the best starts
        # We have budget - len(start_vals) left.
        # Let's distribute this among the top few starts.
        
        # NM options:
        # xatol, fatol, maxiter, adaptive
        
        nm_starts_to_try = min(5, len(start_vals))
        remaining_budget = budget - calls
        
        # Estimate calls per NM run. 
        # NM in dim 5 usually takes ~20-50 calls to converge reasonably.
        # Let's give each NM run a maxiter that corresponds to ~100 calls?
        # Actually, maxiter is number of simplex updates. Each update is 1 call (usually).
        # So maxiter=50 means ~50 calls.
        
        # We want to use the remaining budget.
        # Let's try the best start first with a generous budget.
        
        for i in range(nm_starts_to_try):
            if calls >= budget or time.time() > deadline:
                break
            
            # If we have very little budget left, stop
            if budget - calls < 20:
                break

            v, sp = start_vals[i]
            
            # If the first start was very good, we might want to spend more time polishing it.
            # But let's keep it simple: run NM on each.
            
            # Adaptive NM
            # initial simplex: sp + 0.1 * I (or scaled)
            # Let's use scipy's minimize with method='Nelder-Mead'
            
            # We need to be careful with the simplex size.
            # For f7, the landscape is step-like. A small simplex might get stuck.
            # A large one might jump over the minimum.
            # Default simplex is x0 + 0.05 * |x0| + 0.25 * I.
            # Since x0 is in [-5,5], 0.25 is reasonable.
            
            res = minimize(
                wrapper, 
                sp, 
                method='Nelder-Mead',
                options={
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'maxiter': 100, # ~100 calls
                    'adaptive': True
                }
            )
            
            # Update best if found
            # wrapper already updates best_x, best_f
            
            # Check if we improved significantly
            # If not, maybe try next start
            
        # Step 3: Final Polish with (1+1)-ES
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.05 # Start small
            
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = wrapper(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.9
                    
                if sigma_polish < 1e-5:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
