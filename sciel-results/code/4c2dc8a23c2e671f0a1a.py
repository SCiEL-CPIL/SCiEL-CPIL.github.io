import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return float('inf')
            calls_made += 1
            try:
                fval = func(np.clip(x, -5, 5))
            except Exception:
                return float('inf')
            if not np.isfinite(fval):
                return float('inf')
            return fval
        
        # Phase 1: Global exploration with DE
        pop_size = 20
        de_budget = int(budget * 0.6)
        
        if de_budget > pop_size:
            # Initialize population
            pop = rng.uniform(-5, 5, (pop_size, dim))
            fpop = np.array([safe_func(x) for x in pop])
            
            best_idx = np.argmin(fpop)
            best_x = pop[best_idx].copy()
            best_f = fpop[best_idx]
            
            F = 0.8
            CR = 0.9
            
            max_iter = (de_budget - pop_size) // pop_size
            
            for gen in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                for i in range(pop_size):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    
                    # Select 3 distinct random indices different from i
                    candidates = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(candidates, 3, replace=False)
                    
                    v = pop[a] + F * (pop[b] - pop[c])
                    
                    # Crossover
                    u = v.copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]
                    
                    u = np.clip(u, -5, 5)
                    fu = safe_func(u)
                    
                    if fu < fpop[i]:
                        pop[i] = u
                        fpop[i] = fu
                        if fu < best_f:
                            best_f = fu
                            best_x = u.copy()
                
                # Adapt F and CR every 10 generations
                if gen % 10 == 0 and gen > 0:
                    F = np.clip(F + rng.uniform(-0.2, 0.2), 0.4, 1.0)
                    CR = np.clip(CR + rng.uniform(-0.1, 0.1), 0.5, 0.95)
        
        # Phase 2: Local refinement with (1+1)-ES
        es_budget = budget - calls_made
        if es_budget > 10:
            sigma = 1.0  # Initial step size
            x = best_x.copy()
            f_x = best_f
            success_count = 0
            evals_this_es = 0
            
            # Adapt sigma using 1/5th rule
            while evals_this_es < es_budget and time.time() < deadline:
                # Generate trial point
                direction = rng.normal(0, 1, dim)
                x_trial = x + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = safe_func(x_trial)
                evals_this_es += 1
                
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    success_count += 1
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x_trial.copy()
                
                # Adapt sigma every 5 steps
                if evals_this_es % 5 == 0:
                    success_rate = success_count / 5
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 1e-6, 10.0)
                    success_count = 0
        
        return best_x, best_f
