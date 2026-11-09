import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.rng = np.random.default_rng(42)

    def _eval(self, func, x):
        self.calls += 1
        fx = func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290
        
        # 1. Global Search with Differential Evolution
        pop_size = 10
        # Initialize population
        pop = []
        for _ in range(pop_size):
            x = self.rng.uniform(-5, 5, self.dim)
            f = self._eval(func, x)
            pop.append((x, f))
            
        # Sort by fitness
        pop.sort(key=lambda item: item[1])
        
        F = 0.8
        CR = 0.5
        
        # DE Loop
        # Reserve 50 calls for local search
        local_budget = 50
        de_max_calls = self.budget - local_budget
        
        while self.calls < de_max_calls:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate trial vectors
            # Strategy: best1bin
            x_best = pop[0][0]
            
            new_pop = []
            for i in range(pop_size):
                # Select r1, r2, r3 distinct from i
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(indices, 3, replace=False)
                
                x_r1 = pop[r1][0]
                x_r2 = pop[r2][0]
                x_r3 = pop[r3][0]
                
                # Mutation
                v = x_best + F * (x_r1 - x_r2)
                
                # Crossover
                j_rand = self.rng.integers(0, self.dim)
                trial = np.zeros(self.dim)
                for j in range(self.dim):
                    if self.rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][0][j]
                        
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = self._eval(func, trial)
                
                # Selection
                if f_trial <= pop[i][1]:
                    new_pop.append((trial, f_trial))
                else:
                    new_pop.append(pop[i])
            
            pop = new_pop
            pop.sort(key=lambda item: item[1])

        # 2. Local Refinement
        # Use remaining budget for Nelder-Mead
        remaining_calls = self.budget - self.calls
        if remaining_calls > 0:
            # Wrapper to count calls and check time
            def func_wrapper(x):
                if time.time() > deadline:
                    return self.best_f # Return best known to stop
                return self._eval(func, x)
            
            # Nelder-Mead
            try:
                res = minimize(func_wrapper, self.best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                pass

        return self.best_x, self.best_f
