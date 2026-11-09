import numpy as np
from scipy.optimize import dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5, 5)] * self.dim
        
        # Use dual_annealing which is a global optimizer using simulated annealing
        # and a local optimizer (Nelder-Mead) for refinement.
        # This is distinct from the CMA-ES approach.
        
        res = dual_annealing(
            func,
            bounds=bounds,
            maxfun=self.budget,
            initial_temp=50.0,
            final_temp=1.0,
            seed=42,
            restart_temp_ratio=2e-5
        )
        
        return res.x, res.fun
