import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        limit = 330 - 30  # 300s margin
        
        calls_made = 0
        best_x = np.random.default_rng(0).uniform(-5, 5, self.dim)
        best_f = float('inf')

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Check time
            if time.time() - start_time > limit:
                return best_f
            # Check budget
            if calls_made >= self.budget:
                return best_f
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # Use approx 45% of budget for DE to leave room for local polish
        # popsize=20, maxiter=10 -> 20 * 11 = 220 calls approx
        try:
            res_de = differential_evolution(
                wrapper, 
                bounds=[(-5, 5)] * self.dim, 
                popsize=20, 
                maxiter=10,
                seed=0,
                tol=1e-6,
                init='latinhypercube'
            )
            # Update best
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass

        # Phase 2: Local Polish with Powell
        # Use remaining budget
        # Increase maxiter to utilize remaining budget if DE used less
        try:
            res_powell = minimize(
                wrapper, 
                best_x, 
                method='Powell', 
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res_powell.fun < best_f:
                best_f = res_powell.fun
                best_x = res_powell.x.copy()
        except Exception:
            pass

        return best_x, best_f
