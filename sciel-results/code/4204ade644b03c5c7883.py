import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        N = self.dim
        budget = self.budget
        
        # Phase 1: Sparse Grid Search for Global Basin Identification
        # 3^5 = 243 points. This is deterministic and covers the space [-5, 5]
        # with a step size of 5.0.
        
        grid_points = []
        if budget >= 250:
            # 3 levels: -5, 0, 5
            levels = [-5.0, 0.0, 5.0]
            import itertools
            for combo in itertools.product(levels, repeat=N):
                grid_points.append(np.array(combo))
        else:
            grid_points = [np.zeros(N)]

        # Evaluate grid
        grid_results = []
        try:
            for x in grid_points:
                f_val = evaluate(x)
                grid_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        # Sort grid results by fitness
        grid_results.sort(key=lambda x: x[0])
        
        # Select top K basins
        K = 5
        top_basins = []
        for f_val, x in grid_results[:K]:
            top_basins.append(x)
        
        # If grid search used all budget, we are done.
        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: Local Polish with Nelder-Mead
        def nm_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        for start_x in top_basins:
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
                
            try:
                # Initial simplex with smaller step size for faster convergence
                initial_simplex = [start_x]
                for i in range(N):
                    x_i = start_x.copy()
                    x_i[i] += 0.1
                    initial_simplex.append(x_i)
                
                res = minimize(
                    nm_wrapper, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 500, 
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'initial_simplex': initial_simplex,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        # Phase 3: Final high-precision polish on the best point found
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                # Use L-BFGS-B with tighter tolerances for final polish
                # This is a smooth optimizer that can achieve higher precision
                res = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
