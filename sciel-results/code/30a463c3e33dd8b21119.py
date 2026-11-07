import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                return 1e10
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
        
        rng = np.random.default_rng(12345)
        
        # Initial evaluation to ensure best_x is not None
        x0_init = rng.uniform(-5, 5, n)
        f_init = wrapped_func(x0_init)
        best_f = f_init
        best_x = x0_init
        
        # Multi-Start Nelder-Mead
        while calls < budget and time.time() < deadline - 5:
            # Determine max iterations for this restart
            max_iter = min(50, budget - calls)
            
            if max_iter <= 0:
                break
                
            x0 = rng.uniform(-5, 5, n)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
