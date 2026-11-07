import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x_arr = np.asarray(x, dtype=float).flatten()
                # Ensure x is within bounds, though func clips
                x_arr = np.clip(x_arr, -5.0, 5.0)
                f_val = float(func(x_arr))
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Phase 1: Linear Fit
        x_zero = np.zeros(d)
        f_zero = wrapper(x_zero)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        w_est = np.zeros(d)
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = wrapper(x_probe)
            w_est[i] = (f_probe - f_zero) / 5.0
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Determine optimal corner
        x_corner = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_corner[i] = -5.0
            else:
                x_corner[i] = 5.0
                
        f_corner = wrapper(x_corner)
        
        # Phase 2: Local Search
        # Choose start point: best of zero and corner
        if f_zero < f_corner:
            x_start = x_zero
        else:
            x_start = x_corner
            
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            try:
                # Use Nelder-Mead for robustness against non-linearity
                # maxfev controls the number of function evaluations
                res = scipy.optimize.minimize(
                    wrapper, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 1000, 
                        'maxfev': remaining, 
                        'xatol': 1e-5, 
                        'fatol': 1e-5
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Phase 3: Fill remaining budget
        while calls < budget and time.time() < deadline:
            wrapper(best_x)
            
        return best_x, best_f
