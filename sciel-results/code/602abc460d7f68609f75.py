import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # 1. Random sampling phase to find a good starting point
        # The Bent Cigar function has a very narrow valley. Random sampling helps
        # get closer to the valley than starting at the origin, which might be
        # far from the shifted minimum.
        n_samples = 20
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng()
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
                
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        # 2. Local optimization phase
        # L-BFGS-B is effective for smooth functions like Bent Cigar once in the basin of attraction.
        # We use the best point found in the sampling phase as the initial guess.
        remaining_budget = self.budget - n_samples
        
        try:
            result = minimize(
                func,
                best_x,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0) for _ in range(self.dim)],
                options={'maxiter': 1000, 'maxfev': remaining_budget}
            )
            if result.success or hasattr(result, 'x'):
                return result.x, result.fun
        except Exception:
            pass
            
        return best_x, best_f
