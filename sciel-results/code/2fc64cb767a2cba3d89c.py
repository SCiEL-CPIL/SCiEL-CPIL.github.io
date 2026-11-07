import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290 # 10s margin
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Global Scout (1+1)-ES
        # Start at origin
        x = np.zeros(self.dim)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.5
        
        # Run ES for ~20% of budget
        es_budget = int(self.budget * 0.2)
        
        for _ in range(es_budget - 1):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * np.random.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Slower growth to avoid overshooting
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9 # Slower decay
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
        # Phase 2: Local Refinement with Powell
        if calls < self.budget:
            remaining = self.budget - calls
            
            # Wrapper to track calls and enforce deadline/budget
            # Note: Powell might call func multiple times per iteration.
            # We must ensure we don't exceed self.budget.
            
            def wrapper(x_eval):
                nonlocal calls
                if calls >= self.budget or time.time() > deadline:
                    # Return a large value to signal stop? 
                    # Or raise exception? 
                    # Raising exception might crash the evaluator if not handled.
                    # Better to return current best or a large value.
                    # But Powell expects a float.
                    # If we run out of budget, we should just stop.
                    # However, Powell doesn't have a "stop" callback.
                    # We rely on maxfev.
                    pass
                calls += 1
                return func(x_eval)
            
            # We use maxfev to limit calls.
            # If time runs out, we can't easily stop Powell.
            # But 400 calls is very fast.
            
            try:
                res = minimize(wrapper, best_x, method='Powell', 
                               options={'maxfev': remaining, 'maxiter': 100})
            except Exception:
                pass # Ignore errors in refinement, keep best from scout
            
        return best_x, best_f
