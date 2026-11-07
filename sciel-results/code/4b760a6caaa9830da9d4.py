import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        
        best_x = None
        best_f = np.inf
        call_count = [0]

        def eval_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        # Phase 1: Global Search with Differential Evolution
        # Allocate 60% of budget to DE
        de_budget = int(budget * 0.6)
        
        if time_left() > 10:
            try:
                # DE requires a function that returns a float.
                # maxfev is the max number of function evaluations.
                # popsize is the multiplier for the population size.
                # We use a smaller popsize to allow more iterations within the budget.
                de_result = differential_evolution(
                    eval_func,
                    bounds,
                    maxiter=50, # Max generations
                    maxfev=de_budget, # Hard cap on calls
                    popsize=10, # Smaller population
                    tol=1e-10,
                    seed=42,
                    polish=False, # We will polish manually
                    init='sobol' # Sobol sampling is better for low dims
                )
                if de_result.success or np.isfinite(de_result.fun):
                    if de_result.fun < best_f:
                        best_f = de_result.fun
                        best_x = np.clip(de_result.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 2: Local Search with Powell (Derivative-free, good for Rosenbrock)
        if best_x is not None and call_count[0] < budget and time_left() > 5:
            try:
                powell_result = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if np.isfinite(powell_result.fun) and powell_result.fun < best_f:
                    best_f = powell_result.fun
                    best_x = np.clip(powell_result.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: High-Precision Polish with L-BFGS-B
        if best_x is not None and call_count[0] < budget and time_left() > 2:
            try:
                lbfgs_result = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if np.isfinite(lbfgs_result.fun) and lbfgs_result.fun < best_f:
                    best_f = lbfgs_result.fun
                    best_x = np.clip(lbfgs_result.x, -5.0, 5.0)
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = float(func(best_x))
                call_count[0] += 1
            else:
                best_f = np.inf

        return best_x, best_f
