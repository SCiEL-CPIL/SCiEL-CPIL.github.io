import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            time_limit = 300 - 20
            
            calls_made = 0
            best_x = np.random.default_rng(42).uniform(-5, 5, self.dim)
            best_f = float('inf')
            
            def wrapper(x):
                nonlocal calls_made, best_x, best_f
                if time.time() - start_time > time_limit:
                    return best_f
                if calls_made >= self.budget:
                    return best_f
                
                calls_made += 1
                try:
                    f_val = func(x)
                except Exception:
                    return best_f
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)
                return f_val

            # Phase 1: Coarse Global Search
            # 100 calls
            try:
                res_de1 = differential_evolution(
                    wrapper, 
                    bounds=[(-5, 5)] * self.dim, 
                    popsize=20, 
                    maxiter=5,
                    seed=42,
                    tol=1e-6,
                    init='latinhypercube'
                )
                if res_de1.fun < best_f:
                    best_f = res_de1.fun
                    best_x = np.array(res_de1.x)
            except Exception:
                pass

            # Phase 2: Focused Global Search
            # 100 calls
            # Initialize around the best point found
            if calls_made < self.budget:
                try:
                    # Create initial population around best_x
                    rng = np.random.default_rng(43)
                    # Spread: 1.0 in each dimension
                    init_pop = best_x + rng.uniform(-1.0, 1.0, size=(10, self.dim))
                    # Clip to bounds
                    init_pop = np.clip(init_pop, -5, 5)
                    
                    res_de2 = differential_evolution(
                        wrapper, 
                        bounds=[(-5, 5)] * self.dim, 
                        popsize=10, 
                        maxiter=10,
                        seed=43,
                        tol=1e-6,
                        init=init_pop
                    )
                    if res_de2.fun < best_f:
                        best_f = res_de2.fun
                        best_x = np.array(res_de2.x)
                except Exception:
                    pass

            # Phase 3: Local Polish
            # Remaining budget
            try:
                res_lbfgsb = minimize(
                    wrapper, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'gtol': 1e-12}
                )
                if res_lbfgsb.fun < best_f:
                    best_f = res_lbfgsb.fun
                    best_x = np.array(res_lbfgsb.x)
            except Exception:
                pass

            return best_x, best_f
