import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280 # Safety margin
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # 1. Initial Random Search
        # Use 20 points to find a good region
        initial_points = []
        for _ in range(20):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            initial_points.append((f, x))
            
        # Sort by fitness
        initial_points.sort(key=lambda x: x[0])
        
        # 2. Differential Evolution
        # Population size 10
        pop_size = 10
        # Take best 10 from initial search
        pop = [x for f, x in initial_points[:pop_size]]
        pop_f = [f for f, x in initial_points[:pop_size]]
        
        # If we didn't get 10 points, fill with random
        while len(pop) < pop_size:
             if calls >= budget or time.time() > deadline: break
             x = rng.uniform(-5, 5, dim)
             f = eval_func(x)
             pop.append(x)
             pop_f.append(f)
             
        F = 0.8
        CR = 0.9
        stagnation_counter = 0
        max_stagnation = 10 # Generations
        
        while calls < budget and time.time() < deadline:
            improved = False
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Pick distinct a, b, c != i
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, size=3, replace=False)
                
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)
                
                f_v = eval_func(v)
                
                if f_v < pop_f[i]:
                    pop[i] = v
                    pop_f[i] = f_v
                    improved = True
                    
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if stagnation_counter >= max_stagnation:
                # Restart: Keep best, randomize others
                # Find best in pop
                best_idx = np.argmin(pop_f)
                best_pop_x = pop[best_idx]
                best_pop_f = pop_f[best_idx]
                
                # Re-randomize
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    if calls >= budget or time.time() > deadline:
                        break
                    # Mix of local and global restart
                    if rng.random() < 0.5:
                        x = rng.uniform(-5, 5, dim)
                    else:
                        x = best_pop_x + rng.normal(0, 1.0, dim)
                        x = np.clip(x, -5, 5)
                    f = eval_func(x)
                    pop[i] = x
                    pop_f[i] = f
                stagnation_counter = 0

        return best_x, best_f
