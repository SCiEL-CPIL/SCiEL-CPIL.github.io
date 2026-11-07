import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Initialize
        x = np.random.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls = 1
        
        # Phase 1: (1+5)-ES
        lambda_ = 5
        es_budget = int(self.budget * 0.6)
        
        while calls < es_budget:
            if time.time() > deadline:
                break
                
            offspring = []
            for i in range(lambda_):
                y = x + sigma * np.random.randn(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if time.time() > deadline:
                    break
            
            if len(offspring) < lambda_:
                break
                
            best_offspring = min(offspring, key=lambda t: t[0])
            
            if best_offspring[0] < fx:
                x = best_offspring[1].copy()
                fx = best_offspring[0]
                sigma *= 1.2
            else:
                sigma *= 0.8
                
        # Phase 2: Nelder-Mead Polish
        if calls < self.budget:
            remaining = self.budget - calls
            # NM uses ~N+1 calls per iteration. N=5 -> ~6.
            # Use a safe maxiter
            max_iter = max(1, remaining // 6)
            
            state = {'calls': 0, 'best_x': best_x, 'best_f': best_f}
            
            def wrapped_func(x):
                if time.time() > deadline:
                    return state['best_f']
                state['calls'] += 1
                if state['calls'] > remaining:
                    return state['best_f']
                val = func(x)
                if val < state['best_f']:
                    state['best_f'] = val
                    state['best_x'] = x.copy()
                return val
            
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < state['best_f']:
                    state['best_f'] = res.fun
                    state['best_x'] = res.x
            except Exception:
                pass
                
            best_x = state['best_x']
            best_f = state['best_f']

        return best_x, best_f
