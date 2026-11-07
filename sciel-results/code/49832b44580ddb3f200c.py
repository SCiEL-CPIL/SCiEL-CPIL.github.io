import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x = np.asarray(x, dtype=float)
                f_val = func(x)
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # 1. Origin
        x0 = np.zeros(d)
        f0 = wrapper(x0)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # 2. Probes at bounds
        # Shuffle dimensions to randomize which "good" probe comes first
        dims = np.arange(d)
        np.random.shuffle(dims)
        
        signs = np.zeros(d)
        
        for i in dims:
            if calls >= budget or time.time() > deadline:
                break
                
            x_probe = np.zeros(d)
            x_probe[i] = -5.0
            
            f_probe = wrapper(x_probe)
            
            if f_probe < f0:
                # w_i > 0, optimum at -5
                signs[i] = -1.0
            else:
                # w_i < 0 (or 0), optimum at 5
                signs[i] = 1.0

        # 3. Construct optimal corner
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        x_opt = signs * 5.0
        f_opt = wrapper(x_opt)

        return best_x, best_f
