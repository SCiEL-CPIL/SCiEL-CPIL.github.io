import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        max_time = 310.0
        
        def time_ok():
            return time.time() - start_time < max_time

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return best_f
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # Budget allocation: 60% to DE, 40% to Local
        de_budget = int(budget * 0.6)
        local_budget = budget - de_budget
        
        # DE parameters
        # popsize * dim * maxiter ~ evaluations
        # For dim=5, popsize=10, maxiter=30 -> 1500? No, DE evaluates popsize points per iteration.
        # So popsize * maxiter evaluations.
        # 10 * 30 = 300 evaluations.
        # Let's aim for ~300 evaluations in DE.
        popsize = 10
        maxiter = max(1, de_budget // popsize)
        
        try:
            result_de = differential_evolution(
                wrapped_func,
                bounds=[(-5, 5)] * dim,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                seed=42,
                polish=False # We will do our own polish
            )
            if result_de.fun < best_f:
                best_f = result_de.fun
                best_x = result_de.x.copy()
        except Exception:
            pass

        # Phase 2: Local Refinement with Nelder-Mead
        if time_ok() and calls < budget and best_x is not None:
            remaining_budget = budget - calls
            if remaining_budget > 10:
                try:
                    result_nm = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result_nm.fun < best_f:
                        best_f = result_nm.fun
                        best_x = result_nm.x.copy()
                except Exception:
                    pass

        return best_x, best_f
