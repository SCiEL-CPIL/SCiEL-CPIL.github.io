import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                # Ensure x is within bounds for L-BFGS-B if needed, 
                # but the problem statement says func clips.
                # However, for L-BFGS-B, we pass bounds to the optimizer.
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Global Exploration
        # Sample points in the known minimizer region [-4, 4]^5
        # 10 points is a good start for 5D
        n_explore = 10
        candidates = []
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            # Sample in [-4, 4] to be closer to the minimum
            x = rng.uniform(-4, 4, dim)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x))
        
        if best_x is None:
            return None, None
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization with L-BFGS-B
        # Take the top 3 candidates for local refinement
        top_k = min(3, len(candidates))
        
        # Define bounds for L-BFGS-B
        bounds = [(-5, 5)] * dim
        
        # Allocate remaining budget
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # We will run L-BFGS-B on each of the top_k candidates
        # L-BFGS-B can use many function evaluations to converge.
        # We limit maxfun per run to ensure we stay within budget.
        # A good heuristic is to split the remaining budget among the top runs,
        # but also allow early stopping if convergence is fast.
        
        # Let's reserve a small buffer for safety
        safety_buffer = 10
        effective_remaining = max(0, remaining_budget - safety_buffer)
        
        if effective_remaining <= 0:
            return best_x, best_f
        
        # Run L-BFGS-B on the top candidates
        # We iterate from best to worst. If the best converges quickly, 
        # we have more budget for the next ones.
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = candidates[i][1]
            f0 = candidates[i][0]
            
            # Determine maxfun for this run
            # We want to use up the budget, but not exceed it.
            # L-BFGS-B might stop earlier if it converges.
            # Let's assign a chunk of the remaining budget to this run.
            # If we have 3 runs, maybe 40% to the first, 30% to the second, 30% to the third?
            # Or simply divide equally?
            # Since L-BFGS-B is efficient, let's give it a generous maxfun 
            # but capped by the remaining global budget.
            
            # Simple strategy: use as many as available, but cap at a reasonable number 
            # to avoid wasting time on a non-converging path if it's not the global min.
            # However, for f13, if we start close, it converges fast.
            # Let's just let it use up to the remaining budget for the first one,
            # and if it finishes, use the rest for the next.
            
            # Actually, a better strategy is to run them sequentially, 
            # updating the remaining budget after each.
            
            # maxiter is the max number of iterations (line searches).
            # maxfun is the max number of function evaluations.
            
            # We set maxfun to the remaining budget to allow full convergence 
            # if the global budget is large enough.
            # But we must be careful not to overshoot.
            # The evaluate function will stop if calls >= budget.
            
            # However, L-BFGS-B might not respect the 'evaluate' wrapper's limit 
            # if we pass 'func' directly? No, we pass 'func' which is the raw function.
            # We MUST wrap 'func' to count calls and enforce the limit.
            
            # Let's create a wrapper for L-BFGS-B that uses our evaluate logic
            # but L-BFGS-B calls func(x) directly.
            # So we need to pass a function that calls our evaluate wrapper.
            
            def wrapped_func(x):
                # L-BFGS-B might call this with an array.
                # Our evaluate expects an array.
                val = evaluate(x)
                if val is None:
                    # If budget is exceeded or time out, return a large value 
                    # to signal failure/stop? 
                    # Or raise an exception?
                    # Raising an exception might be cleaner to stop L-BFGS-B.
                    # But L-BFGS-B might catch it?
                    # Better to return a very large number to discourage further search
                    # in that direction, or just stop.
                    # Actually, if we return None, it breaks.
                    # Let's raise a custom exception to break the minimization loop.
                    raise ValueError("Budget exceeded")
                return val
            
            try:
                # Use L-BFGS-B
                # ftol and gtol are convergence criteria
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100, # Max iterations (line searches)
                        'maxfun': remaining_budget, # Max function evals
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                # L-BFGS-B might not update best_x if the last eval was worse
                # We rely on our 'evaluate' wrapper to update best_x
            except Exception as e:
                # If budget exceeded, we just break
                if "Budget exceeded" in str(e):
                    break
                # Other exceptions: ignore and move to next candidate
                pass
            
            # Check time
            if time.time() > deadline:
                break
        
        return best_x, best_f
