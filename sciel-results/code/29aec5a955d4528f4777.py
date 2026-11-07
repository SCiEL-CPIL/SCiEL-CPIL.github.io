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
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search
        # Allocate 30% of budget for global exploration
        n_global = int(budget * 0.3)
        n_global = max(n_global, 30)
        
        # Generate LHS points
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        f_values = np.zeros(n_global)
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # Phase 2: Local Refinement using Powell
        # Powell is often faster than NM for smooth functions in low dimensions
        n_local_candidates = 5
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        budget_per_local = max(10, remaining_budget // n_local_candidates)
        
        call_counter = [0]
        
        def wrapped_func(x):
            if call_counter[0] >= budget - calls or time.time() > deadline:
                return float('inf')
            val = func(x)
            call_counter[0] += 1
            return val

        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = lhs[sorted_indices[k]].copy()
            f0 = f_values[sorted_indices[k]]
            
            try:
                # Powell method
                # maxfev is the maximum number of function evaluations
                # We set it to the allocated budget for this restart
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={
                        'maxfev': budget_per_local,
                        'xrtol': 1e-6,
                        'ftol': 1e-8,
                        'maxiter': 1000
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
            
            calls += call_counter[0]
            call_counter[0] = 0

        return best_x, best_f
