"""Multi-start Nelder-Mead for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart (1+1)-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        def wrapped_func(x):
            nonlocal evals
            if evals >= budget or time_exceeded():
                return np.inf
            try:
                val = func(x)
                evals += 1
                return val
            except Exception:
                evals += 1
                return np.inf

        # Phase 1: Global Exploration
        # Generate 5 initial points
        n_starts = 5
        candidates = []
        for _ in range(n_starts):
            if evals >= budget or time_exceeded():
                break
            x = rng.uniform(-5, 5, size=dim)
            f = wrapped_func(x)
            candidates.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if not candidates:
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement
        # Use Nelder-Mead on the best 3 points
        n_refine = 3
        remaining_budget = budget - evals
        if remaining_budget > 0:
            refine_budget = remaining_budget // n_refine
            
            for i in range(min(n_refine, len(candidates))):
                if evals >= budget or time_exceeded():
                    break
                
                x0 = candidates[i][1]
                
                try:
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': refine_budget, 
                            'maxfev': refine_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 3: Final Polish
        if evals < budget and not time_exceeded():
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': budget - evals, 
                        'maxfev': budget - evals,
                        'xatol': 1e-10,
                        'fatol': 1e-10
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
