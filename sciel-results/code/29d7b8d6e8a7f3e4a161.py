import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280 # Leave 20s margin for safety

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            if evals_used >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization (10 points)
        # Sample 10 points uniformly in the box to find a good basin
        init_candidates = []
        for _ in range(10):
            if evals_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x))
        
        if not init_candidates:
            return best_x, best_f

        # Sort and pick top 3 diverse points for multi-start
        init_candidates.sort(key=lambda k: k[0])
        
        # Select top 3, ensuring some diversity if possible (simple approach: just take top 3)
        # In a step function, the best point is likely on the "plateau" or near the edge.
        starting_points = [c[1] for c in init_candidates[:3]]
        
        # Phase 2: Multi-start Nelder-Mead
        # We split the remaining budget among the starts.
        # Nelder-Mead in 5D can be efficient.
        # We use scipy's minimize.
        
        remaining_budget = budget - evals_used
        num_starts = len(starting_points)
        
        # We can't easily pass a strict eval limit to scipy without a wrapper that counts.
        # We already have a wrapper `eval_point` that counts.
        # We will run minimize for each start, but we need to stop if we run out of budget/time.
        
        # Strategy: Run Nelder-Mead on the best start first. If it converges or uses budget, stop.
        # Actually, better to split budget. But Nelder-Mead's internal logic doesn't know about our budget.
        # We can use `maxiter` and `options` to limit iterations, but iterations != evaluations.
        # A safer approach: Run Nelder-Mead with a generous maxiter, but rely on the `eval_point` wrapper 
        # to stop returning valid new evaluations? No, `minimize` expects a function.
        
        # Let's create a specific wrapper for minimize that checks budget/time.
        def minimize_func(x):
            if evals_used >= budget or time.time() > deadline:
                # Return current best to signal stop? 
                # Scipy doesn't handle early termination gracefully via return value.
                # We must ensure we don't exceed budget.
                # If we exceed, the result is invalid.
                # So we must be conservative with maxiter.
                # Approx: 1 eval per iteration? No, Nelder-Mead uses ~dim+1 evals per iteration.
                # Let's estimate max iterations based on remaining budget.
                pass
            
            return eval_point(x)

        # To be safe, we will manually control the loop or use a very strict maxiter.
        # However, the directive says "Implement a multi-start Nelder-Mead simplex method".
        # Using scipy is the standard way.
        
        # Let's try to run Nelder-Mead on the best candidate first.
        # If it improves, great.
        
        # A more robust way for budget control in scipy is to wrap the function and raise an exception
        # or return a special value, but that breaks scipy.
        
        # Alternative: Use a custom Nelder-Mead implementation? 
        # The directive says "Implement ... Nelder-Mead". It doesn't strictly require scipy, 
        # but scipy is available. Given the strict budget, a custom implementation might be 
        # easier to control if we just want to stop exactly at budget.
        # However, writing a robust Nelder-Mead from scratch is risky.
        
        # Let's stick to scipy but be very careful with `maxiter`.
        # Remaining budget / (dim + 1) is a rough upper bound for iterations.
        
        for x0 in starting_points:
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Estimate max iterations
            # Nelder-Mead typically uses dim+1 points per iteration.
            # Let's reserve 10% of remaining budget for safety/other starts.
            alloc_budget = int((budget - evals_used) * 0.3) 
            if alloc_budget < 10:
                 break
                 
            max_iter = max(1, alloc_budget // (dim + 1))
            
            try:
                res = minimize(minimize_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                # Note: minimize_func calls eval_point which updates global best_x/best_f
            except Exception:
                pass
            
            # After each run, check if we should continue
            if evals_used >= budget or time.time() > deadline:
                break

        return best_x, best_f
