import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
        N = self.dim
        budget = self.budget

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

        # Phase 1: Global Search using Differential Evolution
        # Allocate ~60% of budget to DE.
        popsize = 10
        de_budget = int(budget * 0.6)
        maxiter_de = max(1, de_budget // popsize)
        
        try:
            res_de = differential_evolution(
                evaluate, 
                bounds=[(-5, 5)] * N,
                popsize=popsize,
                maxiter=maxiter_de,
                tol=1e-6,
                seed=0,
                polish=False,
                init='latinhypercube'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x)
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass

        # Phase 2: Local Polish with L-BFGS-B
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    evaluate, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 100,
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
