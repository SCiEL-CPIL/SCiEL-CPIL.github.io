import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        call_count = [0]
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return 1e10
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        bounds = [(-5, 5)] * dim
        
        # Phase 1: Global Search with Differential Evolution
        # Use a small population to conserve budget for polishing
        # popsize=5, maxiter=80 -> 400 evals approx
        try:
            de_result = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=80,
                popsize=5,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
        except Exception:
            de_result = None

        # Phase 2: Local Polish with Nelder-Mead
        if best_x is not None and call_count[0] < budget:
            # Use remaining budget for Nelder-Mead
            # NM in 5D is fast.
            try:
                nm_result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except Exception:
                pass

        if best_x is not None:
            return best_x, best_f
        else:
            # Fallback: return a random point if everything failed (shouldn't happen)
            return np.zeros(dim), 0.0
