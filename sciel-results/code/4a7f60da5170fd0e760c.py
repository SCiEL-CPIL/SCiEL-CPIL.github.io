import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_check():
            return time.time() > deadline

        # Phase 1: Global Exploration
        # Use 40% of budget for random sampling to find a good starting point
        # and to potentially land on the "convex" side of the step.
        exploration_budget = int(self.budget * 0.4)
        
        # Evaluate random points
        # We also include the origin and some structured points
        candidates = [np.zeros(self.dim)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, self.dim))
            
        remaining_explo = exploration_budget - len(candidates)
        if remaining_explo > 0:
            for _ in range(remaining_explo):
                candidates.append(rng.uniform(-5, 5, self.dim))
                
        for x in candidates:
            if calls_made >= self.budget or time_check():
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local Refinement with Powell's Method
        # Powell is derivative-free and robust to non-smoothness (steps).
        # We run it from the best point found.
        if best_x is not None and calls_made < self.budget and not time_check():
            # Wrapper to count calls and enforce deadline
            def powell_func(x):
                nonlocal calls_made
                if time_check():
                    return func(x) # Allow the call to complete, but stop soon
                calls_made += 1
                return func(x)
            
            try:
                # Powell options: 
                # maxiter: max number of iterations
                # maxfev: max number of function evaluations (we control via wrapper, but good to set)
                # xtol, ftol: tolerances
                result = minimize(powell_func, best_x, method='Powell', 
                                  options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8, 'maxfev': self.budget - calls_made})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, try a few more local searches from top candidates
        # We didn't store top candidates in Phase 1 easily without a list.
        # Let's just do a few random restarts of Powell from near best_x.
        if best_x is not None and calls_made < self.budget and not time_check():
            for _ in range(3):
                if calls_made >= self.budget or time_check():
                    break
                # Perturb best_x
                x_start = best_x + 0.2 * rng.standard_normal(self.dim)
                x_start = np.clip(x_start, -5, 5)
                
                def powell_func2(x):
                    nonlocal calls_made
                    if time_check():
                        return func(x)
                    calls_made += 1
                    return func(x)
                
                try:
                    result2 = minimize(powell_func2, x_start, method='Powell',
                                       options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8, 'maxfev': self.budget - calls_made})
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
