import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Deterministic start
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1
        
        delta = 1.0
        min_delta = 1e-7
        
        while calls < self.budget and time.time() < deadline:
            if delta < min_delta:
                break
                
            # Exploratory Phase
            x_exp = x.copy()
            fx_exp = fx
            direction = np.zeros(self.dim)
            
            for i in range(self.dim):
                # Check time/budget
                if calls >= self.budget or time.time() > deadline:
                    break
                    
                # Try +delta
                x_trial = x_exp.copy()
                x_trial[i] += delta
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = func(x_trial)
                calls += 1
                
                if fx_trial < fx_exp:
                    x_exp = x_trial
                    fx_exp = fx_trial
                    direction[i] = delta
                else:
                    # Try -delta
                    if calls >= self.budget or time.time() > deadline:
                        break
                        
                    x_trial = x_exp.copy()
                    x_trial[i] -= delta
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = func(x_trial)
                    calls += 1
                    
                    if fx_trial < fx_exp:
                        x_exp = x_trial
                        fx_exp = fx_trial
                        direction[i] = -delta
            
            # If improvement found in exploratory phase
            if fx_exp < fx:
                x = x_exp
                fx = fx_exp
                
                # Pattern Phase
                x_pat = x + direction
                x_pat = np.clip(x_pat, -5, 5)
                fx_pat = func(x_pat)
                calls += 1
                
                while fx_pat < fx and calls < self.budget and time.time() < deadline:
                    x = x_pat
                    fx = fx_pat
                    x_pat = x + direction
                    x_pat = np.clip(x_pat, -5, 5)
                    fx_pat = func(x_pat)
                    calls += 1
            else:
                # No improvement, reduce step size
                delta *= 0.5
                
        return x, fx
