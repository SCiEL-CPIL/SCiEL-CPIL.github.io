import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                return 1e10 # Force termination
            call_count += 1
            if call_count > self.budget:
                return 1e10
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initialize simplex
        # Center at 0
        x0 = np.zeros(self.dim)
        # Simplex vertices: x0 and x0 + delta * e_i
        delta = 2.0
        simplex = [x0]
        for i in range(self.dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Use Nelder-Mead
        # maxiter controls the number of iterations
        # Each iteration uses roughly dim+1 to 2*dim+2 function calls
        # Budget 500. Dim 5.
        # 500 / 10 = 50 iterations.
        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                           options={'initial_simplex': simplex, 'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        except Exception:
            pass

        return best_x, best_f
