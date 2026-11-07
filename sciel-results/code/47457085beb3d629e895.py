import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        d = self.dim
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Strategy:
        # 1. Try L-BFGS-B from origin.
        # 2. If that uses most of the budget, stop.
        # 3. Use remaining budget for Nelder-Mead polish.
        
        # L-BFGS-B wrapper
        def lbfgsb_func(x):
            if self.calls >= self.budget or time.time() > deadline - 5:
                return 1e10 # Return large value to stop optimization
            f_val = self._evaluate(x)
            if f_val is None:
                return 1e10
            return f_val

        # Initial guess: Origin is usually good for BBOB
        x0 = np.zeros(d)
        
        # Evaluate initial point
        f0 = self._evaluate(x0)
        if f0 is None:
             # If even first eval fails (shouldn't happen), return 0
             return np.zeros(d), 0.0

        # Run L-BFGS-B
        # maxiter is a soft limit; we rely on call count via wrapper
        # finite_diff: True (default for L-BFGS-B if no jac provided)
        try:
            res = minimize(lbfgsb_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * d, 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        # Check if we have budget left for polishing
        if self.calls < self.budget - 10 and time.time() < deadline - 10:
            # Use Nelder-Mead for final polish
            # Nelder-Mead is derivative-free, so it uses ~d+1 calls per step
            # It is good for final convergence.
            
            def nm_func(x):
                if self.calls >= self.budget or time.time() > deadline - 2:
                    return 1e10
                f_val = self._evaluate(x)
                if f_val is None:
                    return 1e10
                return f_val

            x_start = self.best_x if self.best_x is not None else x0
            
            try:
                res2 = minimize(nm_func, x_start, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
            except Exception:
                pass

        # Return the best found
        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
            
        return self.best_x, self.best_f
