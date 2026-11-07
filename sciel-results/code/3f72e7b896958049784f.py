import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # 5s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim

        # Parameters for Multi-Island DE
        num_islands = 5
        pop_per_island = 10
        total_pop = num_islands * pop_per_island
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9 # Crossover rate
        
        # Migration
        migration_interval = 5
        gen_count = 0
        
        # Weierstrass is multimodal, so we start with diverse initial populations
        # Initialize islands
        islands = []
        island_fitness = []
        
        for i in range(num_islands):
            # Random initialization within [-5, 5]
            pop = np.random.uniform(-5, 5, (pop_per_island, dim))
            fitness = np.full(pop_per_island, np.inf)
            
            for j in range(pop_per_island):
                if calls >= max_calls or time.time() > deadline:
                    break
                try:
                    x_eval = pop[j].copy()
                    val = func(x_eval)
                    calls += 1
                    fitness[j] = val
                    if val < best_f:
                        best_f = val
                        best_x = x_eval.copy()
                except:
                    pass
            
            islands.append(pop)
            island_fitness.append(fitness)
            
            if calls >= max_calls or time.time() > deadline:
                break

        # Main DE Loop
        while calls < max_calls and time.time() < deadline:
            gen_count += 1
            
            for i in range(num_islands):
                if calls >= max_calls or time.time() > deadline:
                    break
                    
                pop = islands[i]
                fit = island_fitness[i]
                
                # Sort by fitness
                idx = np.argsort(fit)
                pop = pop[idx]
                fit = fit[idx]
                
                # Identify best in island
                best_idx_i = 0
                best_x_i = pop[best_idx_i]
                best_f_i = fit[best_idx_i]
                
                # Update global best
                if best_f_i < best_f:
                    best_f = best_f_i
                    best_x = best_x_i.copy()
                
                # Generate new population for this island
                new_pop = [pop[best_idx_i]]  # Elitism: keep best
                new_fit = [best_f_i]
                
                for j in range(1, pop_per_island):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    
                    # Select 3 distinct random indices different from j and best
                    # We need r1, r2, r3 != j, r1 != r2 != r3
                    candidates = np.setdiff1d(np.arange(pop_per_island), [j, best_idx_i])
                    if len(candidates) < 3:
                         # Fallback if pop is too small or logic fails, just pick random
                         r1, r2, r3 = np.random.choice(pop_per_island, 3, replace=False)
                    else:
                         r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    
                    # DE/best/1 or DE/rand/1? 
                    # DE/rand/1 is better for exploration, DE/best/1 for exploitation.
                    # Let's use DE/rand/1 for diversity, mixing with best if stagnation.
                    # Actually, standard DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover
                    trial = np.zeros_like(v)
                    j_rand = np.random.randint(0, dim)
                    for k in range(dim):
                        if np.random.rand() <= CR or k == j_rand:
                            trial[k] = v[k]
                        else:
                            trial[k] = pop[j][k]
                    
                    # Clip to bounds (func clips, but we can clip to save time on invalid points? 
                    # No, func clips, so we just evaluate. Clipping ourselves might help if we do local steps later.)
                    # trial = np.clip(trial, -5, 5) # Optional, but func does it.
                    
                    try:
                        val = func(trial)
                        calls += 1
                        if val < fit[j]:
                            new_pop.append(trial)
                            new_fit.append(val)
                        else:
                            new_pop.append(pop[j])
                            new_fit.append(fit[j])
                    except:
                        new_pop.append(pop[j])
                        new_fit.append(fit[j])
                
                # Update island
                islands[i] = np.array(new_pop)
                island_fitness[i] = np.array(new_fit)
                
                # Migration: Every migration_interval generations, replace worst in island i with best from island (i+1)%num_islands
                if gen_count % migration_interval == 0 and i < num_islands - 1:
                    # Find worst in current island
                    idx_worst = np.argmax(island_fitness[i])
                    # Find best in next island
                    next_island = (i + 1) % num_islands
                    idx_best_next = np.argmin(island_fitness[next_island])
                    
                    # Replace
                    islands[i][idx_worst] = islands[next_island][idx_best_next]
                    island_fitness[i][idx_worst] = island_fitness[next_island][idx_best_next]
        
        # Final Polish Phase: Nelder-Mead on the best point found
        # We have remaining budget. Use it to polish best_x.
        remaining_budget = max_calls - calls
        if remaining_budget > 10 and best_x is not None:
            # Wrap func for scipy
            def scipy_func(x):
                nonlocal calls
                if calls >= max_calls or time.time() > deadline:
                    raise ValueError("Stop")
                calls += 1
                return func(x)
            
            try:
                # Nelder-Mead is good for low-dim local optimization
                # Maxiter should be limited to remaining budget
                res = minimize(scipy_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except:
                pass
        elif best_x is None:
             # Fallback if no valid point was found
             best_x = np.zeros(dim)
             if calls < max_calls:
                 try:
                     best_f = func(best_x)
                     calls += 1
                 except:
                     best_f = np.inf

        return best_x, best_f
