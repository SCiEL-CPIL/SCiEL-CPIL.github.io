import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Standard CMA-ES population size
        lambda_pop = 4 + int(3 * np.log2(self.dim))
        
        # Ensure we don't exceed budget
        # Subtract a safety margin for any potential overhead or initialization
        max_iter = int((self.budget - 5) / lambda_pop)
        if max_iter < 1:
            max_iter = 1

        # Start at origin, which is center of [-5, 5]^5
        x0 = np.zeros(self.dim)
        
        # Use CMA-ES
        # tol is set low to encourage exploration until maxiter is hit
        result = minimize(func, x0, method='CMA', 
                          options={'maxiter': max_iter, 'tol': 1e-10, 'disp': False})
        
        return result.x, result.fun
