import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # 300s limit, leave 10s safety margin
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        x_best = None
        f_best = float('inf')
        
        def wrapper(x):
            nonlocal calls_made, f_best, x_best
            if calls_made >= self.budget or time.time() > self.deadline:
                return float('inf')
            
            # Ensure x is within bounds (though scipy might stay in bounds, 
            # Rastrigin is defined everywhere, but task says clip to box)
            # The task says func(x) clips. So we can just pass x.
            
            calls_made += 1
            val = func(x)
            
            if val < f_best:
                f_best = val
                x_best = x.copy()
                
            return val

        # Phase 1: Global Exploration
        n_global = 40
        points = rng.uniform(-5, 5, (n_global, self.dim))
        for p in points:
            val = wrapper(p)
            if val == float('inf'):
                break
        
        if calls_made >= self.budget:
            return x_best, f_best

        # Identify top 4 starting points
        # We need to re-evaluate or keep track. 
        # Since wrapper updates f_best, we lost the individual values for sorting.
        # Let's store them.
        
        # Redo Phase 1 with storage
        # Actually, let's just re-run the logic with storage.
        # To save code, let's just assume we stored the points and values.
        
        # Let's restart the logic cleanly.
        
        # --- Clean Implementation ---
        calls_made = 0
        x_best = None
        f_best = float('inf')
        
        def wrapper(x):
            nonlocal calls_made, f_best, x_best
            if calls_made >= self.budget or time.time() > self.deadline:
                return float('inf')
            calls_made += 1
            val = func(x)
            if val < f_best:
                f_best = val
                x_best = x.copy()
            return val

        # 1. Global Sampling
        n_global = 40
        points = rng.uniform(-5, 5, (n_global, self.dim))
        values = []
        for p in points:
            v = wrapper(p)
            if v == float('inf'):
                values.append(float('inf'))
            else:
                values.append(v)
        
        if calls_made >= self.budget:
            return x_best, f_best

        # 2. Select Top 4
        order = np.argsort(values)
        top_k = 4
        starts = [points[i] for i in order[:top_k]]
        
        # 3. Local Optimization
        remaining = self.budget - calls_made
        # Allocate budget for local search
        # 4 starts, 1 final refinement
        budget_per_local = max(10, remaining // 5)
        
        for s in starts:
            if time.time() > self.deadline or calls_made >= self.budget:
                break
            
            # Use Powell method
            try:
                res = scipy.optimize.minimize(
                    wrapper,
                    s,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': budget_per_local, 'xtol': 1e-8, 'ftol': 1e-8}
                )
            except Exception:
                pass

        # 4. Final Refinement
        if calls_made < self.budget and x_best is not None:
            remaining_final = self.budget - calls_made
            if remaining_final > 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapper,
                        x_best,
                        method='Powell',
                        options={'maxiter': 100, 'maxfev': remaining_final, 'xtol': 1e-10, 'ftol': 1e-10}
                    )
                except Exception:
                    pass

        return x_best, f_best
