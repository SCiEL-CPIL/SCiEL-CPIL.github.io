import numpy as np
from scipy.optimize import basinhopping

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Start from a random point
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        
        def objective(x):
            return func(x)

        # BasinHopping with Nelder-Mead local optimizer
        # niter=10 restarts
        # Each restart uses Nelder-Mead with maxiter=30
        # Total approx 10 * 30 = 300 evals + overhead
        res = basinhopping(
            objective,
            x0,
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'maxiter': 30,
                'xatol': 1e-8,
                'fatol': 1e-8
            },
            niter=10,
            stepsize=2.5,
            seed=42
        )
        
        return res.x, res.fun
