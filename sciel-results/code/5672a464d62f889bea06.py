import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        history = [] # List of (f, x)

        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            calls += 1
            fx = func(x)
            history.append((fx, np.array(x)))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Global Exploration
        # Use random sampling to cover the space better than the previous structured grid.
        # 200 points is a reasonable trade-off between exploration and leaving budget for exploitation.
        rng = np.random.default_rng(42)
        num_explore = 200
        explore_points = rng.uniform(-5.0, 5.0, size=(num_explore, n))
        
        for p in explore_points:
            if calls >= budget or time.time() > deadline - 10:
                break
            wrapped_func(p)

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(n)
            best_f = wrapped_func(best_x)

        # Phase 2: Multi-Start Local Refinement
        # We will attempt to refine the best points found in Phase 1.
        # We collect the top few distinct points from the exploration phase.
        if calls < budget and time.time() < deadline - 5:
            # Get top candidates from history
            # Sort by function value
            history_sorted = sorted(history, key=lambda item: item[0])
            candidates = []
            for f, x in history_sorted:
                # Check if x is distinct from existing candidates
                distinct = True
                for cx in candidates:
                    if np.linalg.norm(x - cx) < 0.5:
                        distinct = False
                        break
                if distinct:
                    candidates.append(x)
                if len(candidates) >= 3:
                    break
            
            for x0 in candidates:
                if calls >= budget or time.time() > deadline - 5:
                    break
                
                remaining = budget - calls
                if remaining < 10:
                    break
                
                # Use Nelder-Mead
                # We limit maxfev to ensure we don't overshoot
                max_fev = min(remaining - 5, 100) # 100 calls per local search
                
                try:
                    def scipy_func(x):
                        return wrapped_func(x)
                    
                    # Use Nelder-Mead with strict limits
                    res = minimize(scipy_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'maxfev': max_fev, 'xatol': 1e-8, 'fatol': 1e-10})
                except Exception:
                    pass

        return best_x, best_f
