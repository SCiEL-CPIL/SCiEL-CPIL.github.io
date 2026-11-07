import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        # Deterministic points
        starts.append(np.zeros(dim))
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 4.0
            starts.append(p)
            p[i] = -4.0
            starts.append(p)
        
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Multi-start Nelder-Mead
        for s in starts:
            if calls[0] >= budget or time.time() > deadline - 10:
                break
            
            # Use scipy's Nelder-Mead for robust local optimization
            # Limit function calls per start to ensure we cover many starts
            # dim=5, NM usually needs ~50-100 calls for good convergence
            # We have 500 budget. 14 starts * 30 calls = 420 calls.
            res = minimize(safe_func, s, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 40})
            
            if res.fun < best_f:
                record(res.x, res.fun)
            
            # If we found a very good point, we might want to polish it more
            # but let's keep moving to find other basins first

        # Final polish on the best found point if budget remains
        if calls[0] < budget and time.time() < deadline - 5:
            # Use remaining budget for a high-precision run on the best point
            remaining = budget - calls[0]
            # Cap the final polish to avoid wasting time if we are already very close
            maxfev_final = min(remaining, 100)
            
            res_final = minimize(safe_func, best_x, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': maxfev_final})
            
            if res_final.fun < best_f:
                record(res_final.x, res_final.fun)

        return best_x, best_f
