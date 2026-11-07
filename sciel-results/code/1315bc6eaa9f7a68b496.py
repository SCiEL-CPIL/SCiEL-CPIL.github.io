import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300
        d = self.dim
        total_budget = self.budget
        
        # Phase 1: Differential Evolution (Global Search)
        # Use 80% of budget for DE
        de_budget = int(total_budget * 0.8)
        popsize = 10
        F = 0.8
        CR = 0.9
        
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (popsize, d))
        fits = np.array([func(x) for x in pop])
        calls = popsize
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        max_generations = (de_budget - calls) // popsize
        
        for gen in range(max_generations):
            if time.time() - start_time > time_limit:
                break
            if calls >= de_budget:
                break
                
            for i in range(popsize):
                if calls >= de_budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                    
                r1, r2, r3 = rng.choice(popsize, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                trial = pop[i].copy()
                j_rand = rng.integers(d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls += 1
                
                if f_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # Phase 2: Nelder-Mead (Local Polish)
        remaining_budget = total_budget - calls
        if remaining_budget > 10: # Need at least a few calls
            # Nelder-Mead uses d+1 calls per iteration
            max_nm_iter = remaining_budget // (d + 1)
            
            # Wrap func to count calls
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls
                if time.time() - start_time > time_limit:
                    return 1e10
                nm_calls += 1
                if nm_calls > remaining_budget:
                    return 1e10
                return func(x)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_nm_iter, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f
