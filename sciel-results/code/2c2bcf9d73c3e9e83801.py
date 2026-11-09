import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution for global search
        pop_size = 20
        de_budget = int(self.budget * 0.8) # 400 calls
        max_gens = de_budget // pop_size
        
        # Initialize population
        population = []
        for i in range(pop_size):
            if i == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-5.0, 5.0, self.dim)
            fx = evaluate(x)
            if fx is None:
                break
            population.append((x, fx))
            
        if not population:
            return best_x, best_f

        # Sort population
        population.sort(key=lambda item: item[1])
        
        F = 0.8
        CR = 0.5
        
        for gen in range(max_gens):
            if calls >= self.budget or time.time() > deadline:
                break
            
            new_population = []
            # Keep the best individual always
            best_ind = population[0]
            new_population.append(best_ind)
            
            for i in range(1, pop_size):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # DE/rand/1/bin mutation
                # Select 3 distinct random individuals different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                x_r1 = population[r1][0]
                x_r2 = population[r2][0]
                x_r3 = population[r3][0]
                x_i = population[i][0]
                
                v = x_r1 + F * (x_r2 - x_r3)
                
                # Crossover
                u = rng.uniform(0, 1, self.dim)
                j_rand = rng.integers(0, self.dim)
                c = np.where((u <= CR) | (np.arange(self.dim) == j_rand), v, x_i)
                
                fc = evaluate(c)
                if fc is None:
                    break
                
                # Selection
                if fc <= population[i][1]:
                    new_population.append((c, fc))
                else:
                    new_population.append(population[i])
            
            if not new_population:
                break
                
            population = new_population
            population.sort(key=lambda item: item[1])
            
            # Stagnation check
            # If best hasn't improved in last 5 generations, restart around best
            # (Simplified: we just continue, DE has some exploration)
            # For better performance, we could restart, but let's keep it simple first.

        # Phase 2: Local Polish with Nelder-Mead
        if best_x is not None and calls < self.budget and time.time() < deadline:
            try:
                # Use a wrapper to count calls and stop if budget exceeded
                remaining_budget = self.budget - calls
                
                def wrapped_func(x):
                    nonlocal calls
                    if calls >= self.budget:
                        return np.inf
                    x = np.clip(x, -5.0, 5.0)
                    fx = float(func(x))
                    calls += 1
                    return fx

                # Nelder-Mead options
                # maxiter is not directly eval count, but we can set a limit
                # In 5D, NM uses ~5-10 evals per iteration.
                # Let's allow up to remaining_budget // 5 iterations
                max_iter = max(1, remaining_budget // 5)
                
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
