import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(777)
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        
        # Initial exploration
        n_init = min(15, budget)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            x = rng.uniform(lo, hi, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if budget <= n_init:
            return best_x, best_f
        
        # (1+1)-ES with adaptive restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.2
        best_ever_f = fx
        best_ever_x = x.copy()
        
        no_improve = 0
        max_no_improve = 20
        min_sigma = 1e-5
        max_sigma = 4.0
        sigma_inc = 1.12
        sigma_dec = 0.88
        
        calls_used = n_init
        remaining = budget - n_init
        
        for step in range(remaining):
            if time.time() > deadline:
                break
            
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, lo, hi)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x = y
                fx = fy
                no_improve = 0
                sigma = min(sigma * sigma_inc, max_sigma)
                
                if fx < best_ever_f:
                    best_ever_f = fx
                    best_ever_x = x.copy()
            else:
                no_improve += 1
                sigma = max(sigma * sigma_dec, min_sigma)
            
            if no_improve >= max_no_improve:
                x = best_ever_x.copy()
                fx = best_ever_f
                no_improve = 0
                
                r = rng.random()
                if r < 0.35:
                    sigma = max(sigma, 2.0)
                elif r < 0.65:
                    sigma = max(sigma, 1.0)
                else:
                    x = rng.uniform(lo, hi, dim)
                    fx = func(x)
                    calls_used += 1
                    if fx < best_ever_f:
                        best_ever_f = fx
                        best_ever_x = x.copy()
                    sigma = 1.0
                    no_improve = 0
                
                if time.time() > deadline:
                    break
        
        return best_ever_x, best_ever_f
