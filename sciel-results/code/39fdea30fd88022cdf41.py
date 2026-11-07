import numpy as np
import scipy.optimize
import time

class Optimizer:
    """Optimizer using Powell's method for derivative-free optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 300.0  # Safety margin before the 330s kill switch
        
        call_count = 0
        limit = self.budget
        
        # Wrapper to enforce budget and time limit
        def wrapped_func(x):
            nonlocal call_count
            if time.time() - start_time > timeout:
                return 1e10
            if call_count >= limit:
                return 1e10
            call_count += 1
            x = np.array(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return float(val)
            except Exception:
                return 1e10

        # Initial guess: Center of the domain is a reasonable heuristic for BBOB
        # which often has the minimum shifted but not to the extreme edges.
        x0 = np.zeros(self.dim)
        
        bounds = [(-5, 5) for _ in range(self.dim)]
        
        best_x = x0
        best_f = wrapped_func(x0)
        
        try:
            # Powell is effective for low-dimensional, smooth, unimodal functions
            # like the BBOB Bent Cigar.
            res = scipy.optimize.minimize(
                wrapped_func, 
                x0, 
                method='Powell', 
                bounds=bounds, 
                options={
                    'maxfev': limit, 
                    'maxiter': 2000, 
                    'disp': False,
                    'ftol': 1e-12,
                    'xtol': 1e-12
                }
            )
            if res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass

        return best_x, best_f
