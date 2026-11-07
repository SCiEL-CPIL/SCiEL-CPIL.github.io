import numpy as np
import time
import scipy.optimize as opt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        d = self.dim
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # We will use scipy's Bobyqa (Bound optimization by quadratic approximation)
        # It is derivative-free and uses a quadratic model, fitting the DF-TBFGS strategy.
        
        # Strategy: Multi-start with Bobyqa
        # d=5, budget=500. 
        # Each Bobyqa run might use ~50-100 evaluations to converge.
        # So we can afford ~5-10 starts.
        
        rng = np.random.default_rng(42)
        
        # Generate starting points
        starts = []
        starts.append(np.zeros(d))
        # Add some structured points and random points
        for _ in range(9):
            starts.append(rng.uniform(-3, 3, d))
        
        for start in starts:
            if calls_used >= max_calls or time.time() > deadline - 5:
                break
            
            # Define a wrapper for scipy that counts calls and checks deadline
            # SciPy's minimize doesn't easily allow stopping mid-iteration based on global state 
            # without a custom callback that raises an exception or returns a specific code.
            # However, we can limit the maxiter.
            
            # Bobyqa parameters:
            # x0: start point
            # bounds: (-5, 5)
            # maxiter: limit iterations to save budget for other starts
            
            # We need to be careful with the budget.
            # Let's allocate a portion of the remaining budget to each start.
            remaining_calls = max_calls - calls_used
            # If we have 500 calls left, and 5 starts left, give each ~100.
            # But Bobyqa might need less.
            # Let's just run it and let the global deadline/budget check inside the wrapper handle it.
            
            # To strictly enforce the call limit within scipy, we can use a callback 
            # that raises an exception if we exceed a local budget or time limit.
            
            local_budget = min(remaining_calls, 100) # Cap per start
            
            def wrapper(x):
                nonlocal calls_used
                if calls_used >= max_calls:
                    raise Exception("Budget Exceeded")
                if time.time() > deadline - 2:
                    raise Exception("Time Exceeded")
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_used += 1
                return fx

            try:
                res = opt.minimize(
                    wrapper, 
                    start, 
                    method='Bobyqa', 
                    bounds=[(-5, 5)] * d, 
                    options={'maxiter': local_budget, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except Exception:
                # If an exception was raised, we stop this start and potentially the whole process
                # if it was due to budget/time.
                if time.time() > deadline - 2 or calls_used >= max_calls:
                    break
                # Otherwise, just continue to next start

        # Final polish if we have time and calls left?
        # Bobyqa is usually quite precise. 
        # If we have a lot of calls left, we can do a final Nelder-Mead or just return.
        
        if best_x is None:
            # Fallback if everything failed
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
