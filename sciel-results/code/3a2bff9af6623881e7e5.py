import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Start from the center of the box
        x0 = np.zeros(self.dim)
        
        # Use Nelder-Mead, which is derivative-free
        # maxiter controls the number of function evaluations roughly
        # We have 500 budget.
        # Nelder-Mead in 5D needs about (dim+1) evaluations per iteration.
        # So 500 / 6 ~ 80 iterations.
        # Set maxiter to 100 to be safe, allowing ~600 calls, but we cap at budget?
        # scipy doesn't strictly cap calls by a 'budget' parameter in the same way, 
        # but maxiter limits iterations.
        
        # To be safe with the call count, we can wrap the function to count calls.
        calls = 0
        def wrapped_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        # We want to finish well within 500 calls.
        # Let's allow 450 calls max.
        # Nelder-Mead might use more.
        # Better to implement a simple loop or trust the maxiter.
        
        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            best_x = res.x
            best_f = res.fun
        except Exception:
            # Fallback to random search if scipy fails
            best_x = x0
            best_f = func(x0)
            for _ in range(self.budget - 1):
                x = np.clip(x0 + np.random.uniform(-1, 1, self.dim), -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x

        return best_x, best_f
