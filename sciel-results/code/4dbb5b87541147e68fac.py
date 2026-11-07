import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        bounds = [(-5, 5)] * d
        
        # Use differential evolution
        # popsize=10 -> 50 individuals
        # maxiter=10 -> 10 generations
        # Total 500 evaluations
        try:
            result = differential_evolution(
                func, 
                bounds=bounds,
                popsize=10, 
                maxiter=10,
                seed=0,
                tol=0.01,
                init='latinhypercube',
                polish=False # Disable local polish to save budget for global search? 
                             # Polish uses Nelder-Mead which might eat budget.
                             # With 500 budget, we want max global coverage.
            )
            return result.x, result.fun
        except Exception:
            # Fallback to simple random search if DE fails
            best_x = np.random.uniform(-5, 5, d)
            best_f = func(best_x)
            for _ in range(self.budget - 1):
                if time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, d)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f
