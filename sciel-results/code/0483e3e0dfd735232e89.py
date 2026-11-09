import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Random Sampling
        n_init = 20
        pop = []
        fvals = []
        for _ in range(n_init):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, d)
            y = call_func(x)
            if y is not None:
                pop.append(x)
                fvals.append(y)
        
        if not pop:
            return best_x, best_y

        # Phase 2: Differential Evolution
        # Use a smaller population to allow more generations
        np_size = min(len(pop), 10) 
        # If we have more than 10, we should probably select the best 10
        if len(pop) > np_size:
            idx = np.argsort(fvals)[:np_size]
            pop = [pop[i] for i in idx]
            fvals = [fvals[i] for i in idx]
        
        # If we have fewer than 10, pad with random or best
        while len(pop) < np_size:
             x = rng.uniform(-5, 5, d)
             y = call_func(x)
             if y is None: break
             pop.append(x)
             fvals.append(y)
        
        pop = np.array(pop)
        fvals = np.array(fvals)
        
        F = 0.8
        CR = 0.5
        
        # Run DE until we have ~30 calls left for local search
        reserve_budget = 30
        while calls_made < budget - reserve_budget and time_ok():
            for i in range(np_size):
                if calls_made >= budget - reserve_budget or not time_ok():
                    break
                
                # Mutation
                r1, r2 = rng.choice(np_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(np_size, 2, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                j_rand = rng.integers(np_size)
                trial = pop[i].copy()
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5, 5)
                y_trial = call_func(trial)
                if y_trial is None:
                    break
                
                # Selection
                if y_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = y_trial
        
        # Phase 3: Local Refinement (Nelder-Mead or simple (1+1)-ES)
        # Use the best point from DE
        if best_x is not None:
            # Simple (1+1)-ES refinement
            sigma = 0.1
            x_cur = best_x.copy()
            y_cur = best_y
            
            # Use remaining budget
            while calls_made < budget and time_ok():
                z = rng.standard_normal(d)
                x_new = x_cur + sigma * z
                x_new = np.clip(x_new, -5, 5)
                y_new = call_func(x_new)
                if y_new is None:
                    break
                
                if y_new < y_cur:
                    x_cur = x_new
                    y_cur = y_new
                    sigma *= 1.1 # Increase step size on success
                else:
                    sigma *= 0.8 # Decrease step size on failure
                    
                if sigma < 1e-6:
                    break

        return best_x, best_y
