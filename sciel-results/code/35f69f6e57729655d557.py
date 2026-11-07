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
        
        n = self.dim
        max_calls = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        def evaluate(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= max_calls:
                return np.inf
            if time.time() > deadline:
                return np.inf
            f_val = func(x)
            total_calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Differential Evolution
        # Reserve some budget for polishing
        polish_budget = 50
        de_budget = max_calls - polish_budget
        
        popsize = 20
        F = 0.8
        CR = 0.9
        
        # Initialize population
        # Include origin and random points
        pop = np.random.uniform(-5, 5, (popsize, n))
        pop[0] = np.zeros(n)
        # Maybe add a few more diverse points
        for i in range(1, min(5, popsize)):
             pop[i] = np.random.uniform(-5, 5, n)
             
        fitness = np.array([evaluate(x) for x in pop])
        
        if total_calls >= de_budget:
            # Not enough budget for main loop, skip to polish
            pass
        else:
            # Main DE loop
            stagnation_count = 0
            prev_best_f = best_f
            
            while total_calls < de_budget:
                if time.time() > deadline:
                    break
                    
                # Check improvement
                if best_f < prev_best_f - 1e-8:
                    prev_best_f = best_f
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                if stagnation_count > 10:
                    # Restart part of population
                    for i in range(popsize):
                        if np.random.rand() < 0.3:
                            pop[i] = np.random.uniform(-5, 5, n)
                            if total_calls < de_budget:
                                fitness[i] = evaluate(pop[i])
                            else:
                                fitness[i] = np.inf
                    stagnation_count = 0
                    continue

                for i in range(popsize):
                    if total_calls >= de_budget:
                        break
                        
                    r1, r2, r3 = np.random.choice(popsize, 3, replace=False)
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    u = np.zeros(n)
                    j_rand = np.random.randint(n)
                    for j in range(n):
                        if np.random.rand() <= CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]
                            
                    u = np.clip(u, -5, 5)
                    f_u = evaluate(u)
                    
                    if f_u <= fitness[i]:
                        pop[i] = u
                        fitness[i] = f_u

        # Phase 2: Polishing with Nelder-Mead
        if total_calls < max_calls:
            # Use the best point found so far
            x0 = best_x.copy()
            
            # Wrap func to count calls
            def count_func(x):
                return evaluate(x)
                
            try:
                res = minimize(count_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
