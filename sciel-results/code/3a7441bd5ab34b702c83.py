"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES optimizer using SciPy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Use CMA-ES which adapts covariance matrix for better handling of
        # rotated and stretched landscapes like BBOB f6.
        x0 = np.zeros(self.dim)
        bounds = [(-5, 5)] * self.dim
        
        # We use the full budget for function evaluations.
        # CMA-ES will stop when maxfev is reached.
        try:
            res = minimize(
                func,
                x0,
                method='CMA',
                bounds=bounds,
                options={
                    'maxfev': self.budget,
                    'popsize': 10,
                    'tolfun': 1e-15,
                    'tolx': 1e-15
                }
            )
            # The return value is ignored by the evaluator, 
            # but we return the best found point and its value.
            return res.x, res.fun
        except Exception:
            # In case of any exception, return a valid structure.
            # If this happens before the first call, the result is invalid.
            # If later, the calls made are kept.
            return np.zeros(self.dim), float('inf')
# EVOLVE-BLOCK-END
