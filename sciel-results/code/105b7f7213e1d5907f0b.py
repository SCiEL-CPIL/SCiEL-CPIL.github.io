import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            # 300s limit, keep 20s buffer for safety
            time_limit = 300 - 20
            
            calls_made = 0
            best_x = np.random.default_rng(42).uniform(-5, 5, self.dim)
            best_f = float('inf')
            
            def wrapper(x):
                nonlocal calls_made, best_x, best_f
                # Time check: Raise exception to abort SciPy loops immediately
                if time.time() - start_time > time_limit:
                    raise TimeoutError("Time limit exceeded")
                # Budget check
                if calls_made >= self.budget:
                    raise RuntimeError("Budget exceeded")
                
                calls_made += 1
                try:
                    f_val = func(x)
                except Exception:
                    return best_f
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)
                return f_val

            # Phase 1: Global Search
            # Use DE. In 5D, popsize 12 is sufficient, saving calls for polish.
            try:
                res_de = differential_evolution(
                    wrapper, 
                    bounds=[(-5, 5)] * self.dim, 
                    popsize=12, 
                    maxiter=25,
                    seed=42,
                    tol=1e-6,
                    init='latinhypercube'
                )
                # best_x and best_f are updated in wrapper
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

            # Phase 2: Local Polish
            # Use L-BFGS-B.
            try:
                res_lbfgsb = minimize(
                    wrapper, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'gtol': 1e-12}
                )
                # best_x and best_f are updated in wrapper
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

            # Phase 3: Final Nelder-Mead Polish
            # Non-gradient method, robust for final steps.
            if calls_made < self.budget and (time.time() - start_time) < time_limit:
                try:
                    res_nm = minimize(
                        wrapper, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    # best_x and best_f are updated in wrapper
                except (TimeoutError, RuntimeError):
                    pass
                except Exception:
                    pass

            return best_x, best_f
